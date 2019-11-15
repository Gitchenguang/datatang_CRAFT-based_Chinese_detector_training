# -*- coding: utf-8 -*-

import cv2
import os
import pdb
import json
import time
import argparse
import numpy as np
import tensorflow as tf
from keras.layers import Input
from keras.models import Model
from net.vgg16 import VGG16_UNet
from collections import OrderedDict
from utils.data_util import load_data
from utils.file_util import list_files, saveResult
from utils.inference_util import getDetBoxes, adjustResultCoordinates
from utils.img_util import load_image, img_resize, img_normalize, to_heat_map

import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()  
config.gpu_options.allow_growth=True  
session = tf.Session(config=config)

KTF.set_session(session)

###########################################################################
# This program compute PPR by doing:
#     1. compute every picture's PPR and num of cheracters
#     2. after done all pictures, compute the PPR of all pics by mutiply weights
#     3. weights is computed by the pic's num divide total num
#
###########################################################################

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/800.h5', type=str, help='pretrained model')
parser.add_argument('--gpu_list', type=str, default='7', help='list of gpu to use')
parser.add_argument('--text_threshold', default=0.5, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.25, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.2, type=float, help='link confidence threshold')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1., type=float, help='image magnification ratio')
parser.add_argument('--show_time', default=True, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default=r'/home/ldf/CRAFT_keras/data/CTW/images-test',
                    type=str, help='folder path to input images')

FLAGS = parser.parse_args()

result_folder = 'results/Cus'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)


def predict(model, image, text_threshold, link_threshold, low_text):
    t0 = time.time()

    # resize
    h, w = image.shape[:2]
    mag_ratio = 800 / max(h, w)
    img_resized, target_ratio = img_resize(image, mag_ratio, FLAGS.canvas_size, interpolation=cv2.INTER_LINEAR)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = img_normalize(img_resized)

    # make score and link map
    score_text, score_link = model.predict(np.array([x]))
    score_text = score_text[0]
    score_link = score_link[0]

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text)
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    white_img = np.ones((render_img.shape[0], 10, 3), dtype=np.uint8) * 255
    ret_score_text = np.hstack((to_heat_map(render_img), white_img, to_heat_map(score_link)))

    if FLAGS.show_time:
        print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, ret_score_text


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou

def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps

def compute_diff(overlaps,gt_boxes,pred_boxes):

    positive_pixel = [1, 2, 3, 4, 5, 6, 7, 8]
    positive_num = np.zeros_like(positive_pixel)
    diff_list = []
    object_num = 0
    
    idx_list = np.argmax(overlaps, 1)
    for i in range(len(idx_list)):
        if overlaps[i, idx_list[i]] <= 0.5:
            continue
        gt_box = gt_boxes[i]
        pred_box = pred_boxes[idx_list[i]]
        diff = [abs(gt_box[c]-pred_box[c]) for c in range(len(gt_box))]
        object_num += 1
        diff_list.append(np.mean(diff))
        for j in range(len(positive_pixel)):
            if diff[0] <= positive_pixel[j] and diff[1] <= positive_pixel[j] and diff[2] <= positive_pixel[j] and diff[3] <= positive_pixel[j]:
                positive_num[j] += 1
    PPR = list()
    if object_num != 0 :
        for j in range(len(positive_pixel)):
            PPR.append(positive_num[j]/object_num)
    else:
        PPR = [0,0,0,0,0,0,0,0]

    return PPR, object_num, diff_list

def compute_final_result(PPR_list,num_list,dif_list):
    final_PPR = [0,0,0,0,0,0,0,0]
    MPD = 0
    for j in range(len(num_list)):
        for i in range(8):
            final_PPR[i] +=  PPR_list[j][i]*num_list[j]/sum(num_list)
        if num_list[j] != 0:
            MPD += dif_list[j]*num_list[j]/sum(num_list)

    return final_PPR, MPD

def flattened_boxes(bboxes):
    bbox = list()
    for i in range(len(bboxes)):        
        bbox.append([bboxes[i][0][0],bboxes[i][1][0],bboxes[i][0][1],bboxes[i][2][1]])
    return bbox

def flattened_boxes_2(char_boxes_list):
    chars = list()
    for i in range(len(char_boxes_list)):
        for j in range(char_boxes_list[i].shape[0]):
            chars.append([min(char_boxes_list[i][j,:,0]),max(char_boxes_list[i][j,:,0]),
                          min(char_boxes_list[i][j,:,1]),max(char_boxes_list[i][j,:,1])])
    return chars    

def compute_PPR(bboxes, char_boxes_list):
    #######################################
    # This part gonna compute PPR for one 
    #   single pic.
    # return:
    #   PPR: a list of PPR from 1 to 8
    #   num: num og total objects
    #######################################

    bboxes_f = flattened_boxes(bboxes)
    char_boxes_list_f = flattened_boxes_2(char_boxes_list)
    if bboxes_f == [] or char_boxes_list_f == [] :
        PPR = [0,0,0,0,0,0,0,0]
        num = 0
        diff = [0,0]
    else:
        cus_overlaps = compute_overlaps(np.array(char_boxes_list_f), np.array(bboxes_f))
        PPR, num, diff = compute_diff(cus_overlaps,char_boxes_list_f,bboxes_f)
    
    return PPR, num, diff

def test():
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list

    """ Load model """
    input_image = Input(shape=(None, None, 3), name='image', dtype=tf.float32)
    region, affinity = VGG16_UNet(input_tensor=input_image, weights=None)
    model = Model(inputs=[input_image], outputs=[region, affinity])
    model.load_weights(FLAGS.trained_model)

    """ For test images in a folder """
    gt_all_imgs = load_data("/home/ldf/CRAFT_keras/data/CTW/gt.pkl") # SynthText, CTW
    t = time.time()
    PPR_list = list()
    num_list = list()
    dif_list = list()
    """ Test images """
    for k, [image_path, word_boxes, words, char_boxes_list] in enumerate(gt_all_imgs):
        
        image = load_image(image_path)
        start_time = time.time()
        bboxes, score_text = predict(model, image, FLAGS.text_threshold, FLAGS.link_threshold, FLAGS.low_text)
        
        """ Compute single pic's PPR and num """
        PPR, single_num, diff = compute_PPR(bboxes, char_boxes_list)
        
        PPR_list.append(PPR)
        num_list.append(single_num)
        dif_list.append(np.mean(diff))

        
    print("elapsed time : {}s".format(time.time() - t))
    result, MPD = compute_final_result(PPR_list,num_list,dif_list)
    print("PPR",result)
    print("MPD",MPD)


if __name__ == '__main__':
    test()
