'''
Coded in 2022/05/29 18:10
Code version: V7
Author: Dachuan Chen
Description:
		This program is coded specifically for 2 types of training image sizes. Most of the 
	txt exporting codes only works when used with proper naming of images.
File Requirement:
	1. test.csv: Filename can only be consisted of number and extention.
	2. test.csv: Can't contain any ' '(space) between filename or type.
	3. test.csv: Need to include 2xxx, 5xxx, 6xxx, 7xxxx image file index.
	4. ./test_images/: Need to include 2xxx, 5xxx, 6xxx, 7xxxx image files.
	5. 2xxx.jpg: Close object train images. Here used for calculating distance of target object. Maximum naming ange within 2001~4999
	6. 5xxx.jpg: Far object train images. Here used for calculating distance of target object. Maximum naming ange within 5001~5999
	7. 6xxx.jpg: Far object train images. Here used for calculating distance of target object. Maximum naming ange within 6001~6999
	8. 7xxx.jpg: Test object images. Maximum naming ange within 7001~inf
Software Requirement:
	1. Python 3.10.4
Functionality:
	Additional Output 增加輸出功能
		1. (csv) Boxed area coordinate 框選範圍座標
		2. (csv) Boxed area center coordinate 框選中心座標
		2. (csv) Boxed area size 框選範圍尺寸
		3. (csv) Boxed area center coordinate difference 框選範圍中心座標變化量
		4. (txt) File info 檔案資訊
		5. (txt) Number count 數量累計
		5. (txt) Accurate number count 精確數量累計
		6. (txt) Boxed area average size (big/small) 框選範圍平均尺寸 (大/小)
		7. (txt) Object size 物體尺寸
		8. (txt) Parameter note 參數備註
	Compress Exported PNG File to 0.5 times of original size.
	Not enough input file error indicator.
Adjustment:
	V6: Add missing data into calculation: far_x_ratio3, far_y_ratio3
	V7: Handel ZeroDivisionError. This error happens when no training image is put into 'test_images' folder.
'''

from __future__ import division
import os
from pickletools import read_bytes1
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
import tensorflow as tf
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras.backend.tensorflow_backend import set_session
from keras_frcnn import roi_helpers

rows = []
coordinate_count0 = 0 #total object
coordinate_count1 = 0 #close object
coordinate_count2 = 0 #far+close object
coordinate_count3 = 0 #far+far+close object
coordinate_count4 = 0 #test+far+far+close object
file_count0 = 0 #total
file_count1 = 0
file_count2 = 0
file_count3 = 0
file_count4 = 0
countera = 0
avg_delta_calculation_boundary1 = 5000 #Image index which devide test images into "trained image" and "test image"
avg_delta_calculation_boundary2 = 6000 #Image index which devide test images into "trained image" and "test image"
avg_delta_calculation_boundary3 = 7000 #Image index which devide test images into "trained image" and "test image"
train_avg_delta_x1 = 1
train_avg_delta_y1 = 1
train_avg_delta_x2 = 1
train_avg_delta_y2 = 1
train_avg_delta_x3 = 1
train_avg_delta_y3 = 1
test_avg_delta_x = 1
test_avg_delta_y = 1
digit_format = "{:.2f}"
close_size = 156
far_size = 400
mix_size = (close_size+far_size)/2
object_identified_flag1 = False
object_identified_flag2 = False
object_identified_flag3 = False
object_identified_flag4 = False
object_unidentified_count0 = 0 #total
object_unidentified_count1 = 0
object_unidentified_count2 = 0
object_unidentified_count3 = 0
object_unidentified_count4 = 0

sys.setrecursionlimit(40000)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

parser = OptionParser()

parser.add_option("-p", "--path", dest="test_path", help="Path to test data.", default='./test_images')
parser.add_option("-n", "--num_rois", type="int", dest="num_rois",
				help="Number of ROIs per iteration. Higher means more memory use.", default=32)
parser.add_option("--config_filename", dest="config_filename", help=
				"Location to read the metadata related to the training (generated when training).",
				default="./train_result/config.pickle")
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='vgg')

(options, args) = parser.parse_args()

if not options.test_path:   # if filename is not given
	parser.error('Error: path to test data must be specified. Pass --path to command line')


config_output_filename = options.config_filename

with open(config_output_filename, 'rb') as f_in:
	C = pickle.load(f_in)

if C.network == 'resnet50':
	import keras_frcnn.resnet as nn
elif C.network == 'vgg':
	import keras_frcnn.vgg as nn

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

img_path = options.test_path

def format_img_size(img, C):
	""" formats the image size based on config """
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape
		
	if width <= height:
		ratio = img_min_side / width
		new_height = int(ratio * height)
		new_width = int(img_min_side)
	else:
		ratio = img_min_side / height
		new_width = int(ratio * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return img, ratio	

def format_img_channels(img, C):
	""" formats the image channels based on config """
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img

def format_img(img, C):
	""" formats an image for model prediction based on config """
	img, ratio = format_img_size(img, C)
	img = format_img_channels(img, C)
	return img, ratio

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):

	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2 ,real_y2)

class_mapping = C.class_mapping

if 'bg' not in class_mapping:
	class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
C.num_rois = int(options.num_rois)

if C.network == 'resnet50':
	num_features = 1024
elif C.network == 'vgg':
	num_features = 512

if K.common.image_dim_ordering() == 'th':
	input_shape_img = (3, None, None)
	input_shape_features = (num_features, None, None)
else:
	input_shape_img = (None, None, 3)
	input_shape_features = (None, None, num_features)


img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)

print(f'Loading weights from {C.model_path}')
model_rpn.load_weights(C.model_path, by_name=True)
model_classifier.load_weights(C.model_path, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

all_imgs = []

classes = {}

bbox_threshold = 0.8

visualise = True

for idx, img_name in enumerate(sorted(os.listdir(img_path))):

	file_count0+=1
	if int(os.path.splitext(img_name)[0]) < avg_delta_calculation_boundary1: #2xxx
		file_count1+=1
	elif int(os.path.splitext(img_name)[0]) < avg_delta_calculation_boundary2: #5xxx
		file_count2+=1
	elif int(os.path.splitext(img_name)[0]) < avg_delta_calculation_boundary3: #6xxx
		file_count3+=1
	else: #7xxx
		file_count4+=1
	if file_count0 != file_count1+file_count2+file_count3+file_count4:
		print('\nfile number error\n')
		break

	if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
		continue
	print(img_name)
	st = time.time()
	filepath = os.path.join(img_path,img_name)

	img = cv2.imread(filepath)

	X, ratio = format_img(img, C)

	if K.common.image_dim_ordering() == 'tf':
		X = np.transpose(X, (0, 2, 3, 1))

	# get the feature maps and output from the RPN
	[Y1, Y2, F] = model_rpn.predict(X)
	

	R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.common.image_dim_ordering(), overlap_thresh=0.7)

	# convert from (x1,y1,x2,y2) to (x,y,w,h)
	R[:, 2] -= R[:, 0]
	R[:, 3] -= R[:, 1]

	# apply the spatial pyramid pooling to the proposed regions
	bboxes = {}
	probs = {}

	for jk in range(R.shape[0] // C.num_rois + 1):

		ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
		if ROIs.shape[1] == 0:
			break

		if jk == R.shape[0] // C.num_rois:
			#pad R
			curr_shape = ROIs.shape
			target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
			ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
			ROIs_padded[:, :curr_shape[1], :] = ROIs
			ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
			ROIs = ROIs_padded

		[P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

		for ii in range(P_cls.shape[1]):

			if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
				continue

			cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

			if cls_name not in bboxes:
				bboxes[cls_name] = []
				probs[cls_name] = []

			(x, y, w, h) = ROIs[0, ii, :]

			cls_num = np.argmax(P_cls[0, ii, :])
			try:
				(tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
				tx /= C.classifier_regr_std[0]
				ty /= C.classifier_regr_std[1]
				tw /= C.classifier_regr_std[2]
				th /= C.classifier_regr_std[3]
				x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
			except:
				pass
			bboxes[cls_name].append([C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h)])
			probs[cls_name].append(np.max(P_cls[0, ii, :]))

	all_dets = []

	for key in bboxes:
		bbox = np.array(bboxes[key])

		new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
		for jk in range(new_boxes.shape[0]):
			(x1, y1, x2, y2) = new_boxes[jk,:]

			(real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

			cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)

			textLabel = f'{key}: {int(100*new_probs[jk])}'
			all_dets.append((key,100 * new_probs[jk]))

			#export x1, y1, x2, y2 coordinate data
			coordinate_count0+=1
			if int(os.path.splitext(img_name)[0]) < avg_delta_calculation_boundary1: #2xxx
				coordinate_count1 = coordinate_count0
				object_identified_flag1 = True
				train_avg_delta_x1 = (train_avg_delta_x1 + (real_x2 - real_x1)) / 2
				train_avg_delta_y1 = (train_avg_delta_y1 + (real_y2 - real_y1)) / 2
				rows.append([img_name,
							 textLabel,
							 digit_format.format(real_x1),
						     digit_format.format(real_y1),
						     digit_format.format(real_x2),
						     digit_format.format(real_y2),
						     digit_format.format((real_x1 + real_x2) / 2),
					         digit_format.format((real_y1 + real_y2) / 2),
					         digit_format.format(real_x2 - real_x1),
					         digit_format.format(real_y2 - real_y1),
					         digit_format.format(real_x2 - real_x1 - train_avg_delta_x1),
				             digit_format.format(real_y2 - real_y1 - train_avg_delta_y1)])

			elif int(os.path.splitext(img_name)[0]) < avg_delta_calculation_boundary2: #5xxx
				coordinate_count2 = coordinate_count0 - coordinate_count1
				object_identified_flag2 = True
				train_avg_delta_x2 = (train_avg_delta_x2 + (real_x2 - real_x1)) / 2
				train_avg_delta_y2 = (train_avg_delta_y2 + (real_y2 - real_y1)) / 2
				rows.append([img_name,
							 textLabel,
							 digit_format.format(real_x1),
						     digit_format.format(real_y1),
						     digit_format.format(real_x2),
						     digit_format.format(real_y2),
						     digit_format.format((real_x1 + real_x2) / 2),
					         digit_format.format((real_y1 + real_y2) / 2),
					         digit_format.format(real_x2 - real_x1),
					         digit_format.format(real_y2 - real_y1),
					         digit_format.format(real_x2 - real_x1 - train_avg_delta_x2),
				             digit_format.format(real_y2 - real_y1 - train_avg_delta_y2)])
			elif int(os.path.splitext(img_name)[0]) < avg_delta_calculation_boundary3: #6xxx
				coordinate_count3 = coordinate_count0 - coordinate_count2 - coordinate_count1
				object_identified_flag3 = True
				train_avg_delta_x3 = (train_avg_delta_x3 + (real_x2 - real_x1)) / 2
				train_avg_delta_y3 = (train_avg_delta_y3 + (real_y2 - real_y1)) / 2
				rows.append([img_name,
							 textLabel,
							 digit_format.format(real_x1),
						     digit_format.format(real_y1),
						     digit_format.format(real_x2),
						     digit_format.format(real_y2),
						     digit_format.format((real_x1 + real_x2) / 2),
					         digit_format.format((real_y1 + real_y2) / 2),
					         digit_format.format(real_x2 - real_x1),
					         digit_format.format(real_y2 - real_y1),
					         digit_format.format(real_x2 - real_x1 - train_avg_delta_x2),
				             digit_format.format(real_y2 - real_y1 - train_avg_delta_y2)])
			else: #7xxx
				coordinate_count4 = coordinate_count0 - coordinate_count3 - coordinate_count2 - coordinate_count1
				object_identified_flag4 = True
				test_avg_delta_x = (test_avg_delta_x + (real_x2 - real_x1)) / 2
				test_avg_delta_y = (test_avg_delta_y + (real_y2 - real_y1)) / 2
				rows.append([img_name,
							 textLabel,
							 digit_format.format(real_x1),
						     digit_format.format(real_y1),
						     digit_format.format(real_x2),
						     digit_format.format(real_y2),
						     digit_format.format((real_x1 + real_x2) / 2),
					         digit_format.format((real_y1 + real_y2) / 2),
					         digit_format.format(real_x2 - real_x1),
					         digit_format.format(real_y2 - real_y1),
					         digit_format.format(real_x2 - real_x1 - test_avg_delta_x),
				             digit_format.format(real_y2 - real_y1 - test_avg_delta_y)])

			(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
			textOrg = (real_x1, real_y1 - 0)

			#cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5),
			#(textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
			#cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5),
			#(textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
			cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

	#print(f'Elapsed time = {time.time() - st})
	#print(all_dets)

	if int(os.path.splitext(img_name)[0]) < avg_delta_calculation_boundary1: #2xxx
		if object_identified_flag1==False: object_unidentified_count1+=1
	elif int(os.path.splitext(img_name)[0]) < avg_delta_calculation_boundary2: #5xxx
		if object_identified_flag2==False: object_unidentified_count2+=1
	elif int(os.path.splitext(img_name)[0]) < avg_delta_calculation_boundary3: #6xxx
		if object_identified_flag3==False: object_unidentified_count3+=1
	else: #7xxx
		if object_identified_flag4==False: object_unidentified_count4+=1

	object_identified_flag1=False
	object_identified_flag2=False
	object_identified_flag3=False
	object_identified_flag4=False
	
	cv2.imwrite('./test_result/{}.png'.format(os.path.splitext(str(img_name))[0]),img, [int(cv2.IMWRITE_PNG_COMPRESSION),9])

print(rows)
a = np.array(rows)
np.savetxt('./test_result/target_coordinate.csv', a, delimiter=',', fmt='%s')

time_tuple = time.localtime()
time_string = time.strftime("%Y/%m/%d, %H:%M:%S", time_tuple)

# Percentage Calculation
try: 
	r1 = (file_count4-object_unidentified_count4)/file_count4
except ZeroDivisionError: 
	r1 = 'Error: file_count4=0 !!ZeroDivisionError!! No test image in folder "test_images". Images are required to generate data.'
try: 
	r2 = (file_count0-object_unidentified_count0)/file_count0
except ZeroDivisionError: 
	r2 = 'Error: file_count0=0 !!ZeroDivisionError!! No image in folder "test_images". Images are required to generate distance data.'
try: 
	r3 = coordinate_count1/file_count1
except ZeroDivisionError: 
	r3 = 'Error: file_count1=0 !!ZeroDivisionError!! No close training image in folder "test_images".'
try: 
	r4 = coordinate_count2/file_count2
except ZeroDivisionError: 
	r4 = 'Error: file_count2=0 !!ZeroDivisionError!! No far1 training image in folder "test_images".'
try: 
	r5 = coordinate_count3/file_count3
except ZeroDivisionError: 
	r5 = 'Error: file_count3=0 !!ZeroDivisionError!! No far2 training image in folder "test_images".'
try: 
	r6 = coordinate_count4/file_count4
except ZeroDivisionError: 
	r6 = 'Error: file_count4=0 !!ZeroDivisionError!! No test image in folder "test_images". Images are required to generate data.'
try: 
	r7 = object_unidentified_count0/file_count0
except ZeroDivisionError: 
	r7 = 'Error: file_count0=0 !!ZeroDivisionError!! No image in folder "test_images". Images are required to generate distance data.'
try: 
	r8 = 1-coordinate_count1/file_count1
except ZeroDivisionError: 
	r8 = 'Error: file_count1=1 !!ZeroDivisionError!! No close training image in folder "test_images".'
try: 
	r9 = 1-coordinate_count2/file_count2
except ZeroDivisionError: 
	r9 = 'Error: file_count2=0 !!ZeroDivisionError!! No far1 training image in folder "test_images".'
try: 
	r10 = 1-coordinate_count3/file_count3
except ZeroDivisionError: 
	r10 = 'Error: file_count3=0 !!ZeroDivisionError!! No far2 training image in folder "test_images".'
try: 
	r11 = 1-coordinate_count4/file_count4
except ZeroDivisionError: 
	r11 = 'Error: file_count4=0 !!ZeroDivisionError!! No test image in folder "test_images". Images are required to generate data.'

# Unidentified Object Count Calculation
object_unidentified_count0 = object_unidentified_count1+object_unidentified_count2+object_unidentified_count3+object_unidentified_count4

# Distance Calculation
try:
	close_x_ratio = test_avg_delta_x/train_avg_delta_x1
except:
	close_x_ratio = 'train_avg_delta_x1=0 ZeroDivisionError'
try:
	close_y_ratio = test_avg_delta_y/train_avg_delta_y1
except:
	close_y_ratio = 'train_avg_delta_y1=0 ZeroDivisionError'
close_xy_ratio = (close_x_ratio+close_y_ratio)/2
avg_close_ratio = (close_x_ratio+close_y_ratio+close_xy_ratio)/3
try:
	far_x_ratio1 = test_avg_delta_x/train_avg_delta_x2
except:
	far_x_ratio1 = 'train_avg_delta_x2=0 ZeroDivisionError'
try:
	far_y_ratio1 = test_avg_delta_y/train_avg_delta_y2
except:
	far_y_ratio1 = 'train_avg_delta_y2=0 ZeroDivisionError'
far_xy_ratio1 = (far_x_ratio1+far_y_ratio1)/2
avg_far_ratio1 = (far_x_ratio1+far_y_ratio1+far_xy_ratio1)/3
try:
	far_x_ratio2 = test_avg_delta_x/train_avg_delta_x3
except:
	far_x_ratio2 = 'train_avg_delta_x3=0 ZeroDivisionError'
try:
	far_y_ratio2 = test_avg_delta_y/train_avg_delta_y3
except:
	far_y_ratio2 = 'train_avg_delta_y3=0 ZeroDivisionError'
far_xy_ratio2 = (far_x_ratio2+far_y_ratio2)/2
avg_far_ratio2 = (far_x_ratio2+far_y_ratio2+far_xy_ratio2)/3
mix_xy_ratio = (close_xy_ratio+far_xy_ratio1+far_xy_ratio2)/3

f = open('./test_result/frcnn_test_info.txt', 'w')

# File Info
f.write('\n===================================================================================\n')
f.write('\nFile Info\n')
f.write('Image Path: ' + img_path + '\n')
f.write('Export Time: ' + time_string + '(UTC+0)' + '\n')
f.write('cols = ["filename", "tag", "x1", "y1", "x2", "y2", "cx", "cy", "Delta x", "Delta y", "delta x diff", "delta y diff"]' + '\n')

# File Count
f.write('\n===================================================================================\n')
f.write('\nFile Count\n')
f.write('Total File Count : ' + str(file_count0) + '\n')
f.write('Total Identified File Count : ' + str(file_count0-object_unidentified_count0) + '\n')
f.write('Total Undentified File Count : ' + str(object_unidentified_count0) + '\n')
f.write('Test File Count(close) : ' + str(file_count1) + '\n')
f.write('Test File Count(far1) : ' + str(file_count2) + '\n')
f.write('Test File Count(far2) : ' + str(file_count3) + '\n')
f.write('Test File Count(test) : ' + str(file_count4) + '\n')

# Percentage
f.write('\n===================================================================================\n')
f.write('\nPercentage\n')
f.write('Test Image Identify Rate: ' + str(r1*100) + '%\n')
f.write('Total Identified Rate : ' + str(r2*100) + '%\n')
f.write('Identified Test File Rate(close) : ' + str(r3*100) + '%\n')
f.write('Identified Test File Rate(far1) : ' + str(r4*100) + '%\n')
f.write('Identified Test File Rate(far2) : ' + str(r5*100) + '%\n')
f.write('Identified Test File Rate(test) : ' + str(r6*100) + '%\n')
f.write('Total Undentified Rate : ' + str(r7*100) + '%\n')
f.write('Undentified Test File Rate(close) : ' + str(r8*100) + '%\n')
f.write('Undentified Test File Rate(far1) : ' + str(r9*100) + '%\n')
f.write('Undentified Test File Rate(far2) : ' + str(r10*100) + '%\n')
f.write('Undentified Test File Rate(test) : ' + str(r11*100) + '%\n')

# Identified Object Count
f.write('\n===================================================================================\n')
f.write('\nIdentified Object Count\n')
f.write('Identified Close Object Count(includes duplicates): ' + str(coordinate_count1) + '\n')
f.write('Identified Far Object Count(includes duplicates): ' + str(coordinate_count2) + '\n')
f.write('Identified Far Object Count(includes duplicates): ' + str(coordinate_count3) + '\n')
f.write('Identified Test Object Count(includes duplicates): ' + str(coordinate_count4) + '\n')
f.write('Identified Total Object Count(includes duplicates): ' + str(coordinate_count0) + '\n')

# Unidentified Object Count
f.write('\n===================================================================================\n')
f.write('\nUnidentified Object Count\n')
f.write('Unidentified Close Object Count(includes duplicates): ' + str(object_unidentified_count1) + '\n')
f.write('Unidentified Far Object Count(includes duplicates): ' + str(object_unidentified_count2) + '\n')
f.write('Unidentified Far Object Count(includes duplicates): ' + str(object_unidentified_count3) + '\n')
f.write('Unidentified Test Object Count(includes duplicates): ' + str(object_unidentified_count4) + '\n')
f.write('Unidentified Total Object Count(includes duplicates): ' + str(object_unidentified_count0) + '\n')

# Delta
f.write('\n===================================================================================\n')
f.write('\nDelta\n')
f.write('Train Image Average Delta X close: ' + str(train_avg_delta_x1) + '\n')
f.write('Train Image Average Delta Y close: ' + str(train_avg_delta_y1) + '\n')
f.write('Train Image Average Delta X far1: ' + str(train_avg_delta_x2) + '\n')
f.write('Train Image Average Delta Y far1: ' + str(train_avg_delta_y2) + '\n')
f.write('Train Image Average Delta X far2: ' + str(train_avg_delta_x3) + '\n')
f.write('Train Image Average Delta Y far2: ' + str(train_avg_delta_y3) + '\n')
f.write('Test Image Average Delta X: ' + str(test_avg_delta_x) + '\n')
f.write('Test Image Average Delta Y: ' + str(test_avg_delta_y) + '\n')

# Distance
f.write('\n===================================================================================\n')
f.write('\nDistance\n')
f.write('Distance calculate with close x data (cm): ' + str(close_x_ratio*close_size) + '\n')
f.write('Distance calculate with close y data (cm): ' + str(close_y_ratio*close_size)+ '\n')
f.write('Distance calculate with close xy data (cm): ' + str(close_xy_ratio*close_size)+ '\n')
f.write('Distance calculate with average close data (cm): ' + str(avg_close_ratio*close_size)+ '\n')
f.write('Distance calculate with far x data1 (cm): ' + str(far_x_ratio1*far_size)+ '\n')
f.write('Distance calculate with far y data1 (cm): ' + str(far_y_ratio1*far_size)+ '\n')
f.write('Distance calculate with far xy data1 (cm): ' + str(far_xy_ratio1*far_size)+ '\n')
f.write('Distance calculate with average far data1 (cm): ' + str(avg_far_ratio1*far_size)+ '\n')
f.write('Distance calculate with far x data2 (cm): ' + str(far_x_ratio2*far_size)+ '\n')
f.write('Distance calculate with far y data2 (cm): ' + str(far_y_ratio2*far_size)+ '\n')
f.write('Distance calculate with far xy data2 (cm): ' + str(far_xy_ratio2*far_size)+ '\n')
f.write('Distance calculate with average far data2 (cm): ' + str(avg_far_ratio2*far_size)+ '\n')
f.write('Distance calculate with mixed xy data (cm): ' + str(mix_xy_ratio*mix_size) + '\n')

# Parameter
f.write('\n===================================================================================\n')
f.write('\nParameter\n')
f.write('Parameters: round_digit: ' + digit_format + '\n')
f.write('Parameters: close_size: ' + str(close_size) + '\n')
f.write('Parameters: far_size: ' + str(far_size) + '\n')
f.write('Parameters: mix_size: ' + str(mix_size) + '\n')
f.write('Parameters: avg_delta_calculation_boundary1: ' + str(avg_delta_calculation_boundary1) + '\n')
f.write('Parameters: avg_delta_calculation_boundary2: ' + str(avg_delta_calculation_boundary2) + '\n')
f.write('Parameters: avg_delta_calculation_boundary3: ' + str(avg_delta_calculation_boundary3) + '\n')

#Variable
f.write('\n===================================================================================\n')
f.write('\nVariable\n')
f.write('file_count1 : ' + str(file_count1) + '\n')
f.write('file_count2 : ' + str(file_count2) + '\n')
f.write('file_count3 : ' + str(file_count3) + '\n')
f.write('file_count4 : ' + str(file_count4) + '\n')
f.write('coordinate_count1 : ' + str(coordinate_count1) + '\n')
f.write('coordinate_count2 : ' + str(coordinate_count2) + '\n')
f.write('coordinate_count3 : ' + str(coordinate_count3) + '\n')
f.write('coordinate_count0 : ' + str(coordinate_count0) + '\n')
f.write('coordinate_count1 : ' + str(coordinate_count1) + '\n')
f.write('coordinate_count2 : ' + str(coordinate_count2) + '\n')
f.write('coordinate_count3 : ' + str(coordinate_count3) + '\n')
f.write('coordinate_count4 : ' + str(coordinate_count4) + '\n')
f.write('coordinate_count0 : ' + str(coordinate_count0) + '\n')
f.write('object_identified_flag1 : ' + str(object_identified_flag1) + '\n')
f.write('object_identified_flag2 : ' + str(object_identified_flag2) + '\n')
f.write('object_identified_flag3 : ' + str(object_identified_flag3) + '\n')
f.write('object_identified_flag4 : ' + str(object_identified_flag4) + '\n')
f.write('object_unidentified_count1 : ' + str(object_unidentified_count1) + '\n')
f.write('object_unidentified_count2 : ' + str(object_unidentified_count2) + '\n')
f.write('object_unidentified_count3 : ' + str(object_unidentified_count3) + '\n')
f.write('object_unidentified_count4 : ' + str(object_unidentified_count4) + '\n')
f.write('object_unidentified_count0 : ' + str(object_unidentified_count0) + '\n')

f.write('\n===================================================================================\n')
#f.write('\n===================================================================================\n')
f.close()
