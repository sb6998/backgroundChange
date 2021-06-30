import os
import pathlib
import matplotlib
import matplotlib.pyplot as plt
import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from six.moves.urllib.request import urlopen
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image
from itertools import compress
import time
tf.get_logger().setLevel('ERROR')

def run_inference(frame):
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()	
	interpreter.set_tensor(input_details[0]['index'], frame)
	interpreter.invoke()	
	results = interpreter.get_tensor(output_details[0]['index'])
	return (results)

vi_path = input("Video Input Path, 0 for camera: ") 

op_img = input("Background Path: ") 
backg = cv2.imread(op_img)
backg = cv2.cvtColor(backg, cv2.COLOR_BGR2RGB)

LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

if vi_path=='0':
	vi_path = int(0)
cap = cv2.VideoCapture(vi_path)##video
#cap = cv2.VideoCapture(0)
h=720
w=1280
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (int(cap.get(3)),int(cap.get(4))))
interpreter = tf.lite.Interpreter("deeplabv3_1_default_1.tflite")
interpreter.allocate_tensors()

while(True):
    ret, frame = cap.read()
    if ret is True:
        opim = backg.reshape((1,720,1280, 3)).astype(np.float32)
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im2 = cv2.resize(im,(257,257), interpolation = cv2.INTER_AREA)
        image_np = im2.reshape((1,257, 257,3)).astype(np.float32)/127.5-1
        image_np_f = im.reshape((1,h,w,3)).astype(np.uint8)
        results = run_inference(image_np)
        seg_map = tf.argmax(tf.image.resize(results, (257, 257)), axis=3)
        seg_map = tf.squeeze(seg_map).numpy().astype(np.int8)
        seg_map = np.where(seg_map==15,255,0)
        seg_map = cv2.resize(seg_map.astype('uint8'),(w,h), interpolation=cv2.INTER_CUBIC)
        inv_mask = np.where(seg_map==255,0,1)
        inv_mask = np.repeat(inv_mask[:,:,np.newaxis],3,axis=2)
        seg_map = np.repeat(seg_map[:,:,np.newaxis],3,axis=2)
        seg_map = cv2.bitwise_and(image_np_f[0].astype('uint8'),seg_map)
        
        opim[0] = np.add(np.multiply(opim[0].astype('uint8'),inv_mask),seg_map)
        out.write(cv2.cvtColor(opim[0].astype('uint8'), cv2.COLOR_BGR2RGB))
        cv2.imshow('img',cv2.cvtColor(opim[0].astype('uint8'), cv2.COLOR_BGR2RGB))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("no frame")
        break

cap.release()
out.release()
cv2.destroyAllWindows()