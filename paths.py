import json
import os
from os.path import join

rootFolder = "artifacts"
image = "image"
answer = "answer"
input_images_folder = join(rootFolder, image)
answer_images_folder = join(rootFolder, answer)
db = "db"
model = "model"
envFileName = "env.json"


def initPaths():
    if not os.path.exists(rootFolder):
        os.mkdir(rootFolder)
    if not os.path.exists(input_images_folder):
        os.mkdir(input_images_folder)
    if not os.path.exists(answer_images_folder):
        os.mkdir(answer_images_folder)


def getEnv():
    if not os.path.exists(envFileName):
        initEnv = {
            "rootFolderName": "artifacts",
            "appAddress": "0.0.0.0",
            "appPort": 8001,
            "dnsHost": "https://detect.dentist.cfu.famedev.ru"
        }
        initEnvJson = json.dumps(initEnv)
        envFile = open(envFileName, "a")
        envFile.write(initEnvJson)
        envFile.close()
    loadedEnvJson = open(envFileName, "r").read()
    env = json.loads(loadedEnvJson)
    return env
