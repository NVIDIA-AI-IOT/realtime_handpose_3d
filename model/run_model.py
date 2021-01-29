'''
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import os
import sys
import torch
import numpy as np

# PROJ ROOT DIR
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(DIR_PATH, os.path.pardir)
sys.path.append(ROOT_PATH)

# Importing Project Library
from pipeline.model_setup import ModelSetup
from pipeline.utils import find_prediction_mask, get_bboxes, find_jaccard_overlap

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_ssd(model_setup:ModelSetup, image: torch.tensor, trt_optim=False):
    """
    Perform inference on the image and return the boundiong boxes along with the images.

    :param model_setup: ModelSetup class instance (holds all teh informatioj about a model)
    :param image: a set of images (N, 1, 300, 300)
    :return: pred_boxes, pred_labels, pred_scores
    """
    model_setup.bb_model.eval()

    model_setup.bb_model.to(DEVICE)
    image = image.to(DEVICE)
    with torch.no_grad():
        pred_locs, pred_scores = model_setup.bb_model(image)
        pred_boxes, pred_labels, pred_scores = model_setup.priors.detect_objects(pred_locs, pred_scores)

    return pred_boxes[0].to("cpu"), pred_labels, pred_scores

def run_centernet(model_setup: ModelSetup, image: torch.tensor, trt_optim=False):
    """
    Run either training or validation on the model

    :param model_setup: ModelSetup, model setup state
    :param train: bool, run training or validation
    """
    model_setup.bb_model.to(DEVICE)
    model_setup.bb_model.eval()
    
    image = image.to(DEVICE)
    with torch.no_grad():
        if trt_optim:
            preds = model_setup.bb_model(image)
        else:
            preds = model_setup.bb_model(image)

    prediction = preds

    pred_heatmap = prediction[0][0:model_setup.centernet_num_classes].max(0)[0].float()
    pred_mask = find_prediction_mask(pred_heatmap)[0][0]
    pred_yx_locations = torch.nonzero(pred_mask)

    pred_height = prediction[0][-4][pred_mask]
    pred_width = prediction[0][-3][pred_mask]

    pred_offset_y = prediction[0][-2][pred_mask]
    pred_offset_x = prediction[0][-1][pred_mask]

    pred_bboxes = get_bboxes(pred_yx_locations, pred_height, pred_width, pred_offset_x, pred_offset_y)

    if pred_bboxes:
        pred_bboxes = torch.FloatTensor(pred_bboxes)
        # Do Non-Max suppression on the nearby boxes
        tmp_boxes = pred_bboxes.clone()
        tmp_boxes[:,2:4] += tmp_boxes[:,0:2]

        # Tensor of zeros for all valid boxes
        suppress = torch.zeros((tmp_boxes.size(0)), dtype=torch.uint8).to(DEVICE)
        # Over lap score [0-1]
        overlap = find_jaccard_overlap(tmp_boxes, tmp_boxes)
        for box in range(tmp_boxes.size(0)):
            if suppress[box] == 1:
                continue
            suppress = torch.max(suppress, torch.as_tensor(overlap[box] > 0.3, dtype=torch.uint8).to(DEVICE))
            suppress[box] = 0

        # Get the list of the valid boxes
        pred_bboxes_list = []
        for i, elem in enumerate(suppress):
            if elem.item() == 0:
                pred_bboxes_list.append(pred_bboxes[i].tolist())
        pred_bboxes = torch.FloatTensor(pred_bboxes_list)
    else:
        pred_bboxes = None
        
    return pred_bboxes, None, None

def run_a2j(model_setup:ModelSetup, image):
    """
    Perform inference on the image and return the boundiong boxes along with the images.

    :param model_setup: ModelSetup class instance (holds all teh informatioj about a model)
    :param image: a set of images (N, 1, 144, 160)
    :return: pred_boxes, pred_labels, pred_scores
    """
    model_setup.a2j_model.eval()
    model_setup.a2j_model.to(DEVICE)
    model_setup.post_process.to(DEVICE)
    image = image.to(DEVICE)

    with torch.no_grad():
        joint_classification, offset_regression, depth_regression = model_setup.a2j_model(image.type(torch.float32))
        pred_points = model_setup.post_process(joint_classification, offset_regression, depth_regression)

    return pred_points
