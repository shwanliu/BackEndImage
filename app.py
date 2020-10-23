#coding: UTF-8
import sys
import faceAttr
import lightOpenPose
import faceSwap
import faceEmotion
from flask import Flask
from flask_cors import *
from tools import *
import json
import collections
import base64
import datetime
import os
import random
import numpy as np
import urllib
import cv2
import argparse

from flask import Flask, render_template, request, jsonify
app = Flask(__name__)
CORS(app, support_credentails=True)

IMAGESERVER = "http://localhost:8082"
FACEATTR_DIR_KEYS = './ImagesTemp/faceAttr/'
POSE_DIR_KEYS = './ImagesTemp/pose/'
FACESWAP_DIR_KEYS = './ImagesTemp/swap/'
FANET_CHECKPOINTSPATH = "./faceAttr/checkpoints/epoch_60FANet.pth"
LIGHTPOSE_CHECKPOINTSPATH = "./lightOpenPose/checkpoints/checkpoint_iter_370000.pth"
EMOTION_CHECKOPINTPATH = './faceEmotion/checkpoints/epoch_99ERNet.pth'

THRESHOLD = 0.6
CLASSNAME = ['刚长出的双颊胡须', '柳叶眉', '吸引人的', '眼袋', '秃头', '刘海', '大嘴唇', '大鼻子', '黑发', '金发', '模糊的', '棕发', '浓眉', '圆胖的', '双下巴', '眼镜', '山羊胡子', '灰发或白发', '浓妆', '高颧骨', '男性', '微微张开嘴巴', '胡子，髭', '细长的眼睛', '无胡子', '椭圆形的脸', '苍白的皮肤', '尖鼻子', '发际线后移', '红润的双颊', '连鬓胡子', '微笑', '直发', '卷发', '戴着耳环', '戴着帽子', '涂了唇膏', '戴着项链', '戴着领带', '年轻人']
LABELS = {'anger': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprised': 6}
print("------------------Load Model Begin------------------")
start = datetime.datetime.now()
FaceAttr = faceAttr.getFaceAttr(FANET_CHECKPOINTSPATH, THRESHOLD,CLASSNAME)
pose = lightOpenPose.getHumenPose(LIGHTPOSE_CHECKPOINTSPATH )
Emotion = faceEmotion.getEmotion(EMOTION_CHECKOPINTPATH, LABELS)
print((datetime.datetime.now() - start))
print("------------------Load Model Over------------------")

@app.route('/')
def hello_world():
  return 'Hello, This is InsightFace!'

@app.route('/123', methods=['POST'])
def upload():
    code = 20000
    status = "done"
    data = "hello,lxy"
    return JsonResult(data,code,status)

@app.route('/faceAttr', methods=['POST'])
def getFaceAttr():
    code = 20000
    status = "done"
    data = ""
    try:
        file = request.files.get('image')
        # data = request.data
            # 写死保存目录，需修改
        filename = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S%f"))+str(random.randint(0,100))+'.jpg'
        filepath = os.path.join(FACEATTR_DIR_KEYS,filename)
        file.save(filepath)
        pred = FaceAttr.prediect(filepath)
        data=pred
    except Exception as ex:
        print(ex)
        code = 500
        status = "unexcept"
    return JsonResult(data,code,status)

@app.route('/getPose', methods=['POST'])
def getPose():
    code = 20000
    status = "done"
    data = ""
    try:
        file = request.files.get('image')
        # data = request.data
            # 写死保存目录，需修改
        imgName = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S%f"))+str(random.randint(0,100))
        filename = imgName +'.jpg'
        filepath = os.path.join(POSE_DIR_KEYS,filename)
        file.save(filepath)
        imageList = []
        imageList.append(filepath)
        frame_provider = ImageReader(imageList)
        savefilename = imgName +'_getPose.jpg'
        savefilepath = os.path.join(POSE_DIR_KEYS,savefilename)
        # file.save(savefilepath)
        processImg = pose.run_demo(frame_provider,265,True,0,0,savefilepath)
        # processImg
        data= IMAGESERVER + processImg.replace("./ImagesTemp",'')
    except Exception as ex:
        print(ex)
        code = 500
        status = "unexcept"
    return JsonResult(data,code,status)

@app.route('/faceSwap', methods=['POST'])
def getFaceSwap():
    code = 20000
    status = "done"
    data = ""
    try:
        print(request)
        src_img = request.files.get('srcImage')
        dst_img = request.files.get('dstImage')
        imgName = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S%f"))+str(random.randint(0,100))
        src_filename = imgName +'_src.jpg'
        dst_filename = imgName +'_dst.jpg'
        out_filename = imgName +'_out.jpg'
        src_filepath = os.path.join(FACESWAP_DIR_KEYS, src_filename )
        dst_filepath = os.path.join(FACESWAP_DIR_KEYS, dst_filename )
        out_filepath = os.path.join(FACESWAP_DIR_KEYS, out_filename )
        # print(src_filepath)
        # print(dst_filepath)
        # print(out_filepath)
        src_img.save(src_filepath)
        dst_img.save(dst_filepath)
        swap = faceSwap.getSwapFace(src_filepath,dst_filepath,out_filepath)
        parser = argparse.ArgumentParser(description='FaceSwapApp')
        parser.add_argument('--warp_2d', default=False, action='store_true', help='2d or 3d warp')
        parser.add_argument('--correct_color', default=False, action='store_true', help='Correct color')
        parser.add_argument('--no_debug_window', default=False, action='store_true', help='Don\'t show debug window')
        args = parser.parse_args()
        processImg = swap.getResult(args)
        data= IMAGESERVER + processImg.replace("./ImagesTemp",'')
    except Exception as ex:
        print(ex)
        code = 500
        status = "unexcept"
    return JsonResult(data,code,status)

@app.route('/getEmotion', methods=['POST'])
def getEmotion():
    code = 20000
    status = "done"
    data = ""
    try:
        file = request.files.get('image')
        # data = request.data
            # 写死保存目录，需修改
        imgName = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S%f"))+str(random.randint(0,100))
        filename = imgName +'.jpg'
        filepath = os.path.join(POSE_DIR_KEYS,filename)
        file.save(filepath)
        data = Emotion.prediect(filepath)
    except Exception as ex:
        print(ex)
        code = 500
        status = "unexcept"
    return JsonResult(data,code,status)

if __name__ == '__main__':
    app.run('0.0.0.0', port=18080, debug=False)
