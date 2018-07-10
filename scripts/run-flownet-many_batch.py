#!/usr/bin/env python2.7
# python run-flownet-many_batch.py /media/sagan/Drive2/sagar/EGTEA_Gaze_Plus/codes/flownet2/models/FlowNet2-Sintel/FlowNet2-CSS-Sintel_weights.caffemodel.h5 /media/sagan/Drive2/sagar/EGTEA_Gaze_Plus/codes/flownet2/models/FlowNet2-Sintel/FlowNet2-CSS-Sintel_deploy.prototxt.template /media/sagan/Drive2/sagar/EGTEA_Gaze_Plus/dataset/flow_net.txt
from __future__ import print_function

import os, sys, numpy as np
sys.path.append('/media/sagan/Drive2/sagar/EGTEA_Gaze_Plus/codes/flownet2/python')
import argparse
from scipy import misc
import caffe
import tempfile
from math import ceil
import cv2
import time

parser = argparse.ArgumentParser()
parser.add_argument('caffemodel', help='path to model')
parser.add_argument('deployproto', help='path to deploy prototxt template')
parser.add_argument('listfile', help='one line should contain paths "img0.ext img1.ext out.flo"')
parser.add_argument('--gpu',  help='gpu id to use (0, 1, ...)', default=1, type=int)
parser.add_argument('--verbose',  help='whether to output all caffe logging', action='store_true')

args = parser.parse_args()

if(not os.path.exists(args.caffemodel)): raise BaseException('caffemodel does not exist: '+args.caffemodel)
if(not os.path.exists(args.deployproto)): raise BaseException('deploy-proto does not exist: '+args.deployproto)
if(not os.path.exists(args.listfile)): raise BaseException('listfile does not exist: '+args.listfile)

def makeColorwheel():

    #  color encoding scheme

    #   adapted from the color circle idea described at
    #   http://members.shaw.ca/quadibloc/other/colint.htm

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3]) # r g b

    col = 0
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0, RY, 1)/RY)
    col += RY

    #YG
    colorwheel[col:YG+col, 0]= 255 - np.floor(255*np.arange(0, YG, 1)/YG)
    colorwheel[col:YG+col, 1] = 255;
    col += YG;

    #GC
    colorwheel[col:GC+col, 1]= 255 
    colorwheel[col:GC+col, 2] = np.floor(255*np.arange(0, GC, 1)/GC)
    col += GC;

    #CB
    colorwheel[col:CB+col, 1]= 255 - np.floor(255*np.arange(0, CB, 1)/CB)
    colorwheel[col:CB+col, 2] = 255
    col += CB;

    #BM
    colorwheel[col:BM+col, 2]= 255 
    colorwheel[col:BM+col, 0] = np.floor(255*np.arange(0, BM, 1)/BM)
    col += BM;

    #MR
    colorwheel[col:MR+col, 2]= 255 - np.floor(255*np.arange(0, MR, 1)/MR)
    colorwheel[col:MR+col, 0] = 255
    return  colorwheel

def computeColor(u, v):

    colorwheel = makeColorwheel();
    nan_u = np.isnan(u)
    nan_v = np.isnan(v)
    nan_u = np.where(nan_u)
    nan_v = np.where(nan_v) 

    u[nan_u] = 0
    u[nan_v] = 0
    v[nan_u] = 0 
    v[nan_v] = 0

    ncols = colorwheel.shape[0]
    radius = np.sqrt(u**2 + v**2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a+1) /2 * (ncols-1) # -1~1 maped to 1~ncols
    k0 = fk.astype(np.uint8)     # 1, 2, ..., ncols
    k1 = k0+1;
    k1[k1 == ncols] = 0
    f = fk - k0

    img = np.empty([k1.shape[0], k1.shape[1],3])
    ncolors = colorwheel.shape[1]
    for i in range(ncolors):
        tmp = colorwheel[:,i]
        col0 = tmp[k0]/255
        col1 = tmp[k1]/255
        col = (1-f)*col0 + f*col1
        idx = radius <= 1
        col[idx] = 1 - radius[idx]*(1-col[idx]) # increase saturation with radius    
        col[~idx] *= 0.75 # out of range
        img[:,:,2-i] = np.floor(255*col).astype(np.uint8)

    return img.astype(np.uint8)


def computeImg(flow):

    eps = sys.float_info.epsilon
    UNKNOWN_FLOW_THRESH = 1e9
    UNKNOWN_FLOW = 1e10

    u = flow[: , : , 0]
    v = flow[: , : , 1]

    maxu = -999
    maxv = -999

    minu = 999
    minv = 999

    maxrad = -1
    #fix unknown flow
    greater_u = np.where(u > UNKNOWN_FLOW_THRESH)
    greater_v = np.where(v > UNKNOWN_FLOW_THRESH)
    u[greater_u] = 0
    u[greater_v] = 0
    v[greater_u] = 0 
    v[greater_v] = 0

    maxu = max([maxu, np.amax(u)])
    minu = min([minu, np.amin(u)])

    maxv = max([maxv, np.amax(v)])
    minv = min([minv, np.amin(v)])
    rad = np.sqrt(np.multiply(u,u)+np.multiply(v,v)) 
    maxrad = max([maxrad, np.amax(rad)])
    #print('max flow: %.4f flow range: u = %.3f .. %.3f; v = %.3f .. %.3f\n' % (maxrad, minu, maxu, minv, maxv))

    u = u/(maxrad+eps)
    v = v/(maxrad+eps)
    img = computeColor(u, v)
    return img
    
def writeFlow(name, flow):
        f = open(name, 'wb')
        f.write('PIEH'.encode('utf-8'))
        np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
        flow = flow.astype(np.float32)
        flow.tofile(f)

def readTupleList(filename):
    list = []
    for line in open(filename).readlines():
        if line.strip() != '':
            list.append(line.split())

    return list

ops = readTupleList(args.listfile)

width = -1
height = -1
ent=ops[0]
num_blobs = 2
input_data = []
img0 = misc.imread(ent[0])
if len(img0.shape) < 3: input_data.append(img0[np.newaxis, np.newaxis, :, :])
else:                   input_data.append(img0[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])
img1 = misc.imread(ent[1])
if len(img1.shape) < 3: input_data.append(img1[np.newaxis, np.newaxis, :, :])
else:                   input_data.append(img1[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])

#print (np.shape(input_data[0]), np.shape(input_data[1]))

if width != input_data[0].shape[3] or height != input_data[0].shape[2]:
    width = input_data[0].shape[3]
    height = input_data[0].shape[2]

    vars = {}
    vars['TARGET_WIDTH'] = width
    vars['TARGET_HEIGHT'] = height

    divisor = 64.
    vars['ADAPTED_WIDTH'] = int(ceil(width/divisor) * divisor)
    vars['ADAPTED_HEIGHT'] = int(ceil(height/divisor) * divisor)

    vars['SCALE_WIDTH'] = width / float(vars['ADAPTED_WIDTH']);
    vars['SCALE_HEIGHT'] = height / float(vars['ADAPTED_HEIGHT']);
    tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)

    proto = open(args.deployproto).readlines()
    for line in proto:
        for key, value in vars.items():
            tag = "$%s$" % key
            line = line.replace(tag, str(value))

        tmp.write(line)

    tmp.flush()

if not args.verbose:
    caffe.set_logging_disabled()
caffe.set_device(args.gpu)
caffe.set_mode_gpu()
net = caffe.Net(tmp.name, args.caffemodel, caffe.TEST)
batch_size=28
for row in range(50000,len(ops),batch_size):
    print (row)
    start_time = time.time()
    ent=ops[row:row+batch_size]
    input_data = []
    input_data_0 = []
    input_data_1 = []
    for ii in range(batch_size):
        #print(row+ii,'Processing tuple:', ent[ii])
        img0 = misc.imread(ent[ii][0])
        input_data_0.append(img0.transpose(2, 0, 1))                 
        img1 = misc.imread(ent[ii][1])
        input_data_1.append(img1.transpose(2, 0, 1))

    input_data_0 = np.asarray(input_data_0)
    input_data_1 = np.asarray(input_data_1)
    input_data = np.asarray([input_data_0,input_data_1])
    #print (input_data[0].shape, input_data[1].shape)

    input_dict = {}
    for blob_idx in range(num_blobs):
        input_dict[net.inputs[blob_idx]] = input_data[blob_idx]
    #print (input_dict)

    # There is some non-deterministic nan-bug in caffe
    #
    #print('Network forward pass using %s.' % args.caffemodel)
    i = 1
    while i<=5:
        i+=1
        net.forward(**input_dict)
        containsNaN = False
        for name in net.blobs:
            blob = net.blobs[name]
            has_nan = np.isnan(blob.data[...]).any()

            if has_nan:
                print('blob %s contains nan' % name)
                containsNaN = True

        if not containsNaN:
            #print('Succeeded.')
            break
        else:
            print('**************** FOUND NANs, RETRYING ****************')
 
    blob = np.squeeze(net.blobs['predict_flow_final'].data).transpose(2, 3, 1,0)
    for ii in range(batch_size):
        img = computeImg(blob[:,:,:,ii]) 
        cv2.imwrite(ent[ii][2], img)

    print("--- %s seconds ---" % (time.time() - start_time))
