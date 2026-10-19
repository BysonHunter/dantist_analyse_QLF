"""Regenerate Figure 3 from QLF-IMG-012; usage: python figure3_revised.py --images DIR --output PNG."""
from pathlib import Path
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from qlf_reanalysis import segment,signals,components

def make(images,output):
 p=next(Path(images).glob('QLF-IMG-012.*'));bgr=cv2.imread(str(p));rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
 mask,_=segment(bgr);F,R,H,E,_,_=signals(bgr,mask)
 contour=rgb.copy();cnt,_=cv2.findContours(mask.astype(np.uint8)*255,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(contour,cnt,-1,(40,240,70),2)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13})
 fig=plt.figure(figsize=(13.5,12.6));gs=fig.add_gridspec(3,5,width_ratios=[1,1,.025,1,.05],hspace=.54,wspace=.15)
 palette=['#ff5935','#ff9130','#ffb930','#ffdc20','#ffff00']
 for row,(v,domain,t0,title,maxv) in enumerate([(F,mask,10,'(a) Green-channel fluorescence loss, ΔF (%)',100),(R,mask,45,'(b) Separate red-channel loss, ΔRloss (intensity units)',150),(H,E,45,'(c) Hybrid score, Δ₁ (intensity units)',150)]):
  binary,metrics=components(v,domain,t0)
  bounds=[10,45,60,75,90,100] if row==0 else [45,60,75,90,120,510]
  # right=True implements lower-open, upper-closed classes matching strict baseline cutoffs.
  idx=np.digitize(v,bounds[1:-1],right=True)
  rgba=ListedColormap(palette)(idx/4)[:,:,:3];overlay=rgb.astype(float)/255
  overlay[domain & ~binary]=.65
  overlay[binary]=.25*overlay[binary]+.75*rgba[binary]
  a=fig.add_subplot(gs[row,0]);a.imshow(contour);a.set_title('Original image and tooth mask',fontsize=13,pad=10)
  a=fig.add_subplot(gs[row,1]);a.imshow(overlay);a.set_title('Candidate-region map',fontsize=13,pad=10)
  a.text(.5,-.08,f"{metrics['regions']} regions; {metrics['area_pct']:.2f}% of "+('E' if row==2 else 'M'),ha='center',va='top',transform=a.transAxes,fontsize=12)
  a=fig.add_subplot(gs[row,3]);im=a.imshow(np.ma.masked_where(~domain,v),cmap='hot',vmin=0,vmax=maxv);a.set_facecolor('#eeeeee');a.set_title('Signal map within '+('E' if row==2 else 'M'),fontsize=13,pad=10)
  cb=fig.colorbar(im,cax=fig.add_subplot(gs[row,4]));cb.ax.tick_params(labelsize=11)
  y=.978-row*.325
  fig.text(.065,y,title,fontsize=15,fontweight='bold',va='top')
  from matplotlib.patches import Patch
  symbol='ΔF' if row==0 else ('ΔRloss' if row==1 else 'Δ₁')
  intervals=['10 < ΔF ≤ 45%','45 < ΔF ≤ 60%','60 < ΔF ≤ 75%','75 < ΔF ≤ 90%','90 < ΔF ≤ 100%'] if row==0 else [f'45 < {symbol} ≤ 60',f'60 < {symbol} ≤ 75',f'75 < {symbol} ≤ 90',f'90 < {symbol} ≤ 120',f'{symbol} > 120']
  handles=[Patch(facecolor='#a6a6a6',label='Not retained as candidate')]+[Patch(facecolor=c,label=t) for c,t in zip(palette,intervals)]
  fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.055,y-.263),ncol=3,frameon=False,fontsize=12,columnspacing=2.4,handlelength=1.5,handleheight=1.0)

 for a in fig.axes:
  if a.get_subplotspec().colspan.start in [0,1,3]:a.axis('off')
 fig.subplots_adjust(left=.04,right=.965,top=.93,bottom=.08)
 fig.savefig(output,dpi=300,facecolor='white');plt.close(fig)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--images',required=True);p.add_argument('--output',default='Figure3_revised.png');a=p.parse_args();make(a.images,a.output)
