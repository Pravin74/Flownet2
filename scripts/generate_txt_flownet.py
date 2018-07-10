import sys
sys.path.append('../config')
from os import listdir
import csv
import os
import EGTEA as DATA
data_dir=DATA.rgb['data_dir']
data_dir_full=DATA.rgb['data_dir_full']
png_dir=DATA.rgb['png_dir']
flow_dir=DATA.rgb['flow_dir']
out_dir = data_dir+'dataset/'
#flow_net = csv.writer(open(out_dir + 'flow_net.txt', 'w'))
folders=listdir(data_dir+png_dir)

for folder in folders:
    imgs=listdir(data_dir+png_dir+folder)
    if not os.path.exists(data_dir+flow_dir+folder):
        os.makedirs(data_dir+flow_dir+folder)
    imgs.sort()
    for i in range(len(imgs)):
        if i== (len(imgs)-1):
            flow_net.writerow([data_dir_full+png_dir+folder+'/'+imgs[i], data_dir_full+png_dir+folder+'/'+imgs[i],  data_dir_full+flow_dir+folder+'/'+imgs[i]])
        else:
            flow_net.writerow([data_dir_full+png_dir+folder+'/'+imgs[i], data_dir_full+png_dir+folder+'/'+imgs[i+1], data_dir_full+flow_dir+folder+'/'+imgs[i]])
        
    
#### left over video clips   #################
# flow_net = csv.writer(open(out_dir + 'flow_net_leftovers.txt', 'w'))
# left_over=[]
# for folder in folders:
#     imgs=listdir(data_dir+png_dir+folder)
#     imgs_flow=listdir(data_dir+flow_dir+folder)
#     if len(imgs_flow) != len(imgs):
#         print (len(imgs), len(imgs_flow))
#         for i in range(len(imgs)):
#             if i== (len(imgs)-1):
#                 flow_net.writerow([data_dir_full+png_dir+folder+'/'+imgs[i], data_dir_full+png_dir+folder+'/'+imgs[i],  data_dir_full+flow_dir+folder+'/'+imgs[i]])
#             else:
#                 flow_net.writerow([data_dir_full+png_dir+folder+'/'+imgs[i], data_dir_full+png_dir+folder+'/'+imgs[i+1], data_dir_full+flow_dir+folder+'/'+imgs[i]])
#                 