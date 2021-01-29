'''
Copyright (c) 2019 Boshen Zhang
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
from pyk4a import PyK4A

import os
import sys
import time
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# PROJ ROOT DIR
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(DIR_PATH, os.path.pardir)
sys.path.append(ROOT_PATH)

# PROJ LIBRARY
import pipeline.constants as const
from pipeline.utils import *
from pipeline.model_setup import ModelSetup
from model.run_model import run_centernet, run_ssd, run_a2j

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_arguments():
    """
    Argument parser function for main.py
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--trt',
                        type=bool,
                        default=False,
                        help="Set to True for trt optimization")

    args = parser.parse_args()
    return args

def run_camera_inferance(k4a, model_setup: ModelSetup, iterations=100, show_heatmap=False, trt_optim=False):
    """
    Run the model for N number of frames

    :param model_setup: ModelSetup
    :param iterations: the total number of frames to run the model
    :param show_heatmap: set to visualize prediction heat map and mask
    """
    fig = plt.figure(figsize=(6, 8))
    fig.suptitle(f"{const.NUM_JOINTS} Joints", fontsize=16)
    ax_1 = fig.add_subplot(2,1,1)
    ax_2 = fig.add_subplot(2, 1, 2, projection='3d')

    bb_summary = Summary()
    a2j_summary = Summary()

    for i in range(1000):
        capture = k4a.get_capture()
        ir_img = capture.ir
        depth_img = capture.depth

        w, h = ir_img.shape[1], ir_img.shape[0] # Image (width, height)
        transformed_image = centernet_img_transform(ir_image=ir_img, depth_image=depth_img) # Image transfered to (1, 1, 300, 300) float tensor

        start_time = time.time()
        pred_boxes, _, _ = run_centernet(model_setup, transformed_image) # Perform Inference
        end_time = time.time()
        bb_summary.update(end_time-start_time)

        pred_joints_collections = []
        median_depths = []

        if pred_boxes != None:
            # Normalizing the pred boxes to original dimentions
            original_dims = torch.FloatTensor([w, h, w, h]).unsqueeze(0)
            pred_boxes[:,2:4] += pred_boxes[:,0:2]
            pred_boxes /= 320

            pred_boxes *= original_dims

            bboxs = [] # list of (x0, y0, x1, y1)
            for i in range(pred_boxes.size(0)):
                box_locs = pred_boxes[i].tolist()
                x, y = box_locs[0], box_locs[1]
                width, height = abs(box_locs[0] - box_locs[2]), abs(box_locs[1] - box_locs[3])
                rect = patches.Rectangle((x,y),width,height,linewidth=1,edgecolor='g',facecolor='none')
                ax_1.add_patch(rect)

                bboxs.append([
                    int(box_locs[0]),
                    int(box_locs[1]),
                    int(box_locs[2]),
                    int(box_locs[3])
                ])        

            for bbox in bboxs:
                t_depth_image, median_depth = a2j_depth_image_transform(depth_img, bbox)
                # import pdb; pdb.set_trace()
                start_time = time.time()
                pred_points = run_a2j(model_setup, t_depth_image)
                end_time = time.time()
                a2j_summary.update(end_time-start_time)

                pred_joints_collections.append(pred_points[0])
                median_depths.append(median_depth)

            normalized_joints = back_to_normal(pred_joints_collections, bboxs, median_depths)
            scats = vizualize_frams(ax_2, normalized_joints)
                        
        ir_img[ir_img > 3000] = ir_img.mean()
        ax_1.imshow(ir_img, interpolation='nearest', cmap ='gray')

        plt.draw()
        plt.pause(0.001)

        ax_1.clear()
        if pred_boxes != None:
            [scat.remove() for scat in scats]
        ax_2.clear()
        
        print(f"BB Infrence time: {bb_summary.avg:1.4f}	"\
                f"A2J Infrence time: {a2j_summary.avg:1.4f}	"\
                f"Total Infrence time: {a2j_summary.avg + bb_summary.avg:1.4f}")

        print(f"BB Infrence time FPS: {1/bb_summary.avg:1.0f}	"\
                f"A2J Infrence time FPS: {1/a2j_summary.avg:1.0f}	"\
                f"Total Infrence time FPS: {1/(a2j_summary.avg + bb_summary.avg):1.0f}")


def main():
    # Load camera with default config
    k4a = PyK4A()
    k4a.start()
    
    args = parse_arguments()
    bbox_path, a2j_path = get_model()
    
    model_setup = ModelSetup(BBOX_MODEL_PATH=bbox_path, A2J_model_path=a2j_path, trt_optim=args.trt)
    
    run_camera_inferance(k4a, model_setup)

main()
