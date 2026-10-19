"""Reproducible descriptive reanalysis; no diagnostic or treatment decisions.

Usage: python qlf_reanalysis.py --images PATH --output results
Dependencies: numpy scipy pandas matplotlib opencv-python-headless
"""
from pathlib import Path
import argparse, hashlib, json, ast, sys, platform
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COMMIT='90ca8f1ea53dcea5715cf3feac8f9aa568ed0212'
STRUCT=ndi.generate_binary_structure(2,1)

def segment(bgr):
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    hm=cv2.inRange(hsv,np.array((0,0,38),np.uint8),np.array((255,155,255),np.uint8))
    gray=cv2.cvtColor(cv2.bitwise_and(bgr,bgr,mask=hm),cv2.COLOR_BGR2GRAY)
    _,th=cv2.threshold(gray,20,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    closed=cv2.morphologyEx(th,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8),iterations=4)
    contours,_=cv2.findContours(closed,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    mask=np.zeros(bgr.shape[:2],np.uint8)
    cv2.drawContours(mask,contours,-1,255,-1)
    labels,n=ndi.label(mask>0,STRUCT);sizes=np.bincount(labels.ravel());keep=sizes>=500;keep[0]=False
    mask=ndi.binary_fill_holes(keep[labels])
    return mask, {'hsv_pixels':int((hm>0).sum()),'closed_pixels':int((closed>0).sum()),'mask_pixels':int(mask.sum())}

def means(bgr,mask,legacy=False):
    pixels=bgr[mask]
    if legacy:
        flat=pixels[pixels!=0]
        return np.array([flat[k::3].mean() for k in range(3)]) if len(flat) else np.zeros(3)
    pixels=pixels[np.any(pixels!=0,axis=1)]
    return pixels.astype(float).mean(axis=0) if len(pixels) else np.zeros(3)

def components(values,domain,threshold,minimum=20):
    lab,n=ndi.label((values>threshold)&domain,STRUCT)
    sizes=np.bincount(lab.ravel());keep=sizes>=minimum;keep[0]=False
    binary=keep[lab];ids=np.flatnonzero(keep)
    total=int(binary.sum())
    return binary,dict(regions=len(ids),area_px=total,area_pct=100*total/int(domain.sum()) if domain.any() else np.nan,
                       mean_signal=float(values[binary].mean()) if total else np.nan,
                       integrated_signal=float(values[binary].sum()) if total else 0.)

def signals(bgr,mask,legacy=False):
    ref=means(bgr,mask,legacy)
    # Preserve the supplied notebook's float32 arithmetic and strict > cutoff.
    g32=bgr[:,:,1].astype(np.float32)
    fg=np.percentile(g32[mask],95)
    if fg < 1e-3: fg=1.0
    g=g32.astype(float)
    F=np.clip((1-g32/fg)*100,0,100)
    R=np.maximum(0,ref[2]-bgr[:,:,2].astype(float)).astype(np.float32)
    R[~np.any(bgr!=0,axis=2)]=0  # Preserve the notebook's non-black-pixel rule.
    H=R+np.maximum(0,ref[1]-g)
    # Recreate the anti-aliased mask used inside the supplied tooth_Analise.
    cnt,_=cv2.findContours(mask.astype(np.uint8)*255,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    aa=np.zeros(mask.shape,np.uint8);cv2.drawContours(aa,cnt,-1,255,-1,cv2.LINE_AA)
    href=means(bgr,aa>0,legacy)
    hybrid=np.maximum(0,href[1]-g)+np.maximum(0,href[2]-bgr[:,:,2].astype(float))
    eligible=(aa>0)&np.all(bgr!=0,axis=2)&((bgr[:,:,0]<=href[0])|(g<=href[1]))
    return F,R,hybrid,eligible,ref,href

def overlap(a,b):
    u=np.count_nonzero(a|b);return np.count_nonzero(a&b)/u if u else 1.

def main(images,output):
    images=Path(images);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    baseline=[];sensitivity=[];filters=[];audit=[];manifest=[];qrows=[];maps={};failures=[]
    for p in sorted(images.glob('QLF-IMG-*')):
        bgr=cv2.imread(str(p)); assert bgr is not None,p
        mask,diag=segment(bgr);ident=p.stem
        manifest.append(dict(image_id=ident,file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),width=bgr.shape[1],height=bgr.shape[0]))
        if not mask.any():
            failures.append(dict(image_id=ident,status='empty_segmentation',**diag));continue
        F,R,H,eligible,ref,href=signals(bgr,mask)
        _,oldR,oldH,oldE,oldref,oldhref=signals(bgr,mask,True)
        mixed=int(np.count_nonzero(mask&np.any(bgr==0,axis=2)&np.any(bgr!=0,axis=2)))
        audit.append(dict(image_id=ident,mask_pixels=int(mask.sum()),partly_zero_pixels=mixed,
             old_B=oldref[0],old_G=oldref[1],old_R=oldref[2],new_B=ref[0],new_G=ref[1],new_R=ref[2],
             reference_max_abs_change=float(abs(ref-oldref).max()),hybrid_reference_max_abs_change=float(abs(href-oldhref).max())))
        n=int(eligible.sum());q={'image_id':ident,'eligible_pixels':n,'mask_pixels':int(mask.sum())}
        for suffix,hh,ee in [('corrected',H,eligible),('legacy',oldH,oldE)]:
            nn=int(ee.sum());v=hh[ee]
            q['q0_'+suffix]=float(np.mean(v>45)) if nn else np.nan
            for j,(lo,hi) in enumerate([(45,60),(60,75),(75,90),(90,120)],1):q[f'q{j}_{suffix}']=float(np.mean((v>lo)&(v<=hi))) if nn else np.nan
            q['q5_'+suffix]=float(np.mean(v>120)) if nn else np.nan
        qrows.append(q)
        for name,values,domain,t0,grid in [
            ('green',F,mask,10,[5,7.5,10,12.5,15,20]),
            ('red',R,mask,45,[30,35,40,45,50,55,60]),
            ('hybrid',H,eligible,45,[30,35,40,45,50,55,60,75,90,120])]:
            base,m=components(values,domain,t0)
            baseline.append(dict(image_id=ident,method=name,threshold=t0,minimum_px=20,domain_pixels=int(domain.sum()),**m))
            for th in grid:
                binary,mm=components(values,domain,th)
                sensitivity.append(dict(image_id=ident,method=name,threshold=th,minimum_px=20,iou_with_baseline=overlap(base,binary),**mm))
            for minimum in [5,10,20,50,100]:
                binary,mm=components(values,domain,t0,minimum)
                filters.append(dict(image_id=ident,method=name,threshold=t0,minimum_px=minimum,iou_with_baseline=overlap(base,binary),**mm))
            if name=='red':
                oldmask,om=components(oldR,mask,t0)
                audit[-1].update(old_red_area_pct=om['area_pct'],new_red_area_pct=m['area_pct'],old_red_regions=om['regions'],new_red_regions=m['regions'])
        if ident=='QLF-IMG-012':maps={'bgr':bgr,'mask':mask,'F':F,'R':R,'H':H,'eligible':eligible}
        print(ident,'done',flush=True)
    dfs={'baseline':baseline,'threshold_sensitivity':sensitivity,'component_sensitivity':filters,'reference_audit':audit,'image_manifest':manifest,'hybrid_fractions':qrows,'failures':failures}
    for name,rows in dfs.items():pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    b=pd.DataFrame(baseline);s=pd.DataFrame(sensitivity)
    summaries={}
    for name,g in b.groupby('method'):
        summaries[name]={}
        for col in ['regions','area_pct','mean_signal']:
            v=g[col].to_numpy();summaries[name][col]=dict(n=int(np.isfinite(v).sum()),mean=float(np.nanmean(v)),median=float(np.nanmedian(v)),q1=float(np.nanpercentile(v,25)),q3=float(np.nanpercentile(v,75)),minimum=float(np.nanmin(v)),maximum=float(np.nanmax(v)),total=float(np.nansum(v)))
    meta=dict(repository='https://github.com/BysonHunter/dantist_images',commit=COMMIT,
              python=platform.python_version(),opencv=cv2.__version__,numpy=np.__version__,
              images_attempted=len(manifest),images_successful=len(audit),failures=failures,summary=summaries,
              note='Descriptive file-level analysis. Acquisition modality, patient independence and clinical truth not established. Hybrid denominator is the eligible subset; green/red denominator is the tooth mask.')
    (out/'summary.json').write_text(json.dumps(meta,indent=2))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    ids=sorted(b.image_id.unique())
    for ax,col,ylabel in zip(axs,['area_pct','regions'],['Threshold-classified area (%)','Connected candidate regions']):
        for ident in ids:
            v=b[b.image_id==ident].set_index('method');ax.plot([0,1],[v.loc['green',col],v.loc['red',col]],'-o',alpha=.55,lw=.8,ms=3,color='#385c80')
        ax.set_xticks([0,1],['Green ΔF','Separate red ΔR']);ax.set_ylabel(ylabel)
    fig.tight_layout();fig.savefig(out/'paired_channels.png',dpi=220);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    for ax,name in zip(axs,['green','red','hybrid']):
        z=s[s.method==name]
        for _,g in z.groupby('image_id'):ax.plot(g.threshold,g.area_pct,color='#8aa5bc',alpha=.35,lw=.7)
        g=z.groupby('threshold').area_pct;med=g.median();ax.plot(med.index,med.values,'-o',color='#153d5c',lw=2)
        ax.fill_between(med.index,g.quantile(.25).values,g.quantile(.75).values,color='#8aa5bc',alpha=.4)
        ax.set_title(name.capitalize());ax.set_xlabel('Threshold (%)' if name=='green' else 'Threshold (intensity units)');ax.set_ylabel('Classified domain area (%)')
    fig.tight_layout();fig.savefig(out/'threshold_sensitivity.png',dpi=220);plt.close(fig)
    if maps:
        fig,axs=plt.subplots(1,4,figsize=(14,3.7));axs[0].imshow(cv2.cvtColor(maps['bgr'],cv2.COLOR_BGR2RGB));axs[0].set_title('QLF-IMG-012')
        for ax,key,title,t0 in zip(axs[1:],['F','R','H'],['Green ΔF','Separate red ΔR','Hybrid Δ₁'],[10,45,45]):
            domain=maps['eligible'] if key=='H' else maps['mask'];binary,_=components(maps[key],domain,t0)
            ax.imshow(np.ma.masked_where(~binary,maps[key]),cmap='magma',vmin=0,vmax=100 if key=='F' else 150);ax.set_facecolor('#eeeeee');ax.set_title(title)
        for ax in axs:ax.axis('off')
        fig.tight_layout();fig.savefig(out/'illustrative_maps.png',dpi=220);plt.close(fig)
    return meta

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--images',required=True);p.add_argument('--output',default='results');a=p.parse_args();main(a.images,a.output)
