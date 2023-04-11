"""
1217 选择测试集的所有数据进行测试，取平均值
1229 按照论文用康达进行测试  batch值应为1
"""
import argparse
import sys
import re
import numpy as np
sys.path.append('image_comp')
import torch
from torch.autograd import Variable
import configparser
import os
import math
from metric import *
import imageio
import cv2
import torch.utils.data as data
import datasetDistribute0318 as datasetDistribute
import dataset_decoder 
import networkDistribute_pyramid as networkDistribute
from torchvision import transforms
import analysis_side as side_net

def get_args(filename,section):
    args = {}
    config = configparser.RawConfigParser()
    config.read(filename)
    options = config.options(section)
    print(len(options))
    for t in range(len(options)):
        if config.get(section,options[t] ).isdigit():
            args[options[t]] = int(config.get(section,options[t] ))
        else:
            try:
                float(config.get(section,options[t] ))
                args[options[t]] = float(config.get(section,options[t] ))
            except:
                args[options[t]] = config.get(section, options[t])
    return args

def encode_key(filename,QP):
    if os.path.exists('str.bin'):
        os.system('rm str.bin')
        os.system('rm rec.yuv')
        print('delete old str.bin and rec.yuv !')
    
    if os.path.exists('str.bin'):
        print('error')

    if os.path.exists('log.txt'):
        os.system('rm log.txt')
        print('delete old log.txt !')
    print('encodering Key frames ......')
    commd = '/home/user2/HM-HM-16.20+SCM-8.8/bin/TAppEncoderStatic -c key_cfg/encoder_intra_main_rext.cfg -c key_cfg/BQSquare.cfg  -i {} -ts 2 -f 149 -q {} > log.txt'.format('yuv/hall_qcif.yuv',26)
    print('asd:',os.getcwd())
    if os.system(commd)!=0:
        print('encodering Key frames fail !')


    print("Key frames encoder finish !")


    return bps
if(sys.argv[1] == 'encoder'):
    args = get_args("config.ini",'encoder')
    #load_model(args['model'])
    #yuv2img(args['filename'],args['img_path'],144, 176,0)
    #print('Encodering wz frames......')
    #encoder_img(args['test'],args['max_batch'],encoder,binarizer,args['flag'],args['level'],args['iterations'],args['output_name'])
    #print('WZ frames encoder finish, file wzall.npz !')
    encode_key(args['filename'],args['key_qp'])
    #print("encoder bitrate : {} kbps".format(get_bps('str.bin','wzall.npz')))
