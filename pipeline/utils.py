'''
Copyright (c) 2019 Boshen Zhang
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import os
import sys
import cv2
import torch
import numpy as np
import torchvision.transforms.functional as FT

from glob import glob
from PIL import Image, ImageOps

# PROJ ROOT DIR
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(DIR_PATH, os.path.pardir)
sys.path.append(ROOT_PATH)

# PROJ LIBRARY
import pipeline.constants as const


# Set the global device variable to cuda is GPU is avalible
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# DATASET INFO
MEAN = -0.66877532
STD = 28.32958208


def xy_to_cxcy(xy):
    """
    Convert bounding boxes from boundary coordinates (x_min, y_min, x_max, y_max) to center sized coordinates (c_x, c_y, w, h)

    :param xy: bounding box coordinate a tensor of size (n_boxes, 4)
    :return: bounding boxes in bouindary coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([ (xy[:, 2:] + xy[:, :2])/2, # c_x, c_y 
                        xy[:, 2:] - xy[:, :2]], 1) # w, h
            
def cxcy_to_xy(cxcy):
    """
    Convert bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)

    :param cxcy: bounding boxes in center-size coordinate (n_boxes, 4)
    :return: bounding boxes in boundary coordinates (n_boxes, 4)
    """

    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2), # x_min, y_min
                        cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1) # x_max, y_max

def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """
    Encode boundiong boxes (that are in center-size form) w.r.t the corresponding prior boxes.

    For the center coordinates, find the offset with respect to the prior box, and scale by the size of the prior box
    For the size coordinates, scale by the size of teh prior box, and convert to the log-space.

    In the model, we are predicting bounding box coordinates in this encoded form.

    :param cxcy: bounding boxes in center sized coordinates, (n_priors, 4)
    :param priors_xcxy: prior boxes with respect which the encoding must be preformed, (n_priors, 4)
    :return: encoded boundin boxes, (n_priors, 4)
    """
    cxcy = cxcy.to(DEVICE)
    priors_cxcy = priors_cxcy.to(DEVICE)
    return torch.cat(
                    [(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10), # g_c_x, g_c_y
                    torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1) # g_w, g_h

def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """
    Decode bounding box coordinates predicted by thr model, sice they are encoded in the form mentioned above.

    They are decoded into center size coordinates.

    This is invers of the above functions

    :param gcxgcy: encoded bounding box (i.e. output of model) (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding is defined (n_priors, 4)
    :return: decoded bounding boxes in center size form (n_priors, 4)
    """
    gcxgcy = gcxgcy.to(DEVICE)
    priors_cxcy = priors_cxcy.to(DEVICE)
    return torch.cat(
        [gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2], # c_x, c_y
            torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1) # w, h

def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination betweeen 2 sets of boxes that are in boundary coordinates.

    :param set_1: set_1 (n1, 4)
    :param set_2: set 2 (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the set 2 (n1, n2)
    """

    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1).to(DEVICE), set_2[:, :2].unsqueeze(0).to(DEVICE)) # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1).to(DEVICE), set_2[:, 2:].unsqueeze(0).to(DEVICE)) # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0) # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1] # (n1, n2)

def find_jaccard_overlap(set_1, set_2):
    """
    Find IoU of every box combination in between the 2 sets (boxes in boundary coordinates)

    :param set_1: set 1 (n1, 4)
    :param set2: set 2 (n2, 4)
    :return: Jaccard overlap of each of the boxes in the set 1 with respect to set 2 (n1, n2)
    """

    intersection = find_intersection(set_1, set_2)

    area_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1]) # (n1)
    area_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1]) # (n1)

    union = area_set_1.unsqueeze(1).to(DEVICE) + area_set_2.unsqueeze(0).to(DEVICE) - intersection # (n1, n2)

    return intersection / union

def decay_lr_rate(optim, scale):
    """
    Scale the lr rate by a factor.

    :param optim: optimizer (SGD)
    :param scale: factor to scale the lr rate with.
    """
    for param_group in optim.param_groups:
        param_group['lr'] = param_group['lr'] * scale

class Summary(object):
    def __init__(self):
        self.item = 0
        self.sum = 0
        self.len = 0
        self.avg = 0.000001
    
    def update(self, value):
        self.item = value
        self.sum += value
        self.len += 1
        self.avg = self.sum / self.len


def get_model():
    """
    The model weights are saved in CHECKPOINT_DIR specified in constants.py
    this functions loos into that directory and returns the path to the model.

    Please set the correct paths in pipeline/constans.py if not using default:
        SSD_MODEL_PATH
        SSD_DATASET_NAME
        SSD_MODEL_NAME

        A2J_BACKBONE_NAME
        A2J_MODEL_PATH
    
    :return: str, str: path to SSD model, path to A2J model
    """
    centernet_model_path = const.CENTERNET_MODEL_PATH
    bb_models = glob(f"{centernet_model_path}/{const.CENTERNET_LOSS}_{const.CENTERNET_MODEL_NAME}_{const.CENTERNET_DATA_LOADER}.pth")
    if not bb_models:
        print(f"
There are no CenterNet Model check points at:\
                
{centernet_model_path}\
                
Please Train a model or change the directory on constants.py
")
        exit(-1)

    backbone_name = [elem[0] for idx, elem in enumerate(const.A2J_BACKBONE_NAME.items()) if elem[1]][0]
    a2j_model_path = const.A2J_MODEL_PATH
    a2j_model_path = f"{a2j_model_path}/{const.DATASET}_{const.DATA_SEGMENT}_{backbone_name}_{const.NUM_JOINTS}_a2j.pth"
    a2j_models = glob(a2j_model_path)

    if not a2j_models:
        print(f"
There are no A2J Model with {const.A2J_BACKBONE_NAME} backbone in check points at:\
                
{a2j_model_path}\
                
Please Train a model or change the directory on constants.py
")
        exit(-1)    
    
    return bb_models[0], a2j_models[0]

# Image Transforms
def normalize(image: np.array, img_shape=tuple):
    """
    Resize image to (300, 300)

    :param image: numpy array
    :return: normalized image Casted to torch
    """
    image = cv2.resize(image, img_shape, interpolation=cv2.INTER_NEAREST)
    mean = np.mean(image)
    std = image.std()
    if std==0:
        std = 1 
    new_image = (image - mean) / std

    # Cast to pytorch and expand dimentions for the model forward pass    
    new_image = torch.from_numpy(new_image).type(torch.float32)

    new_image = new_image.unsqueeze(0)

    return new_image

def centernet_img_transform(depth_image: np.array, ir_image: np.array, img_shape=const.INPUT_IMG_SIZE, input_type=const.CENTERNET_DATA_LOADER):
    """
    Transform the images into the input images

    :param depth_img: np.array (uint16), depth image
    :param ir_image: np.array (uint16), depth image
    :param img_shape: tuple (h, w), image size
    :param input_type: str, with input type (Fused, Depth) is the model using
    :return: 
    """
    def depth_input(depth_image: np.array, **kwargs):
        c, h, w = depth_image.size()
        depth_image = depth_image[0:1,:,:] # depth
        new_image = depth_image.expand(1, 3, h, w)
        return new_image

    def fused_input(depth_image: np.array, ir_image:np.array):
        c, h, w = depth_image.size()
        new_image_c1 = depth_image.type(torch.float32)
        new_image_c2 = ir_image.type(torch.float32)
        new_image_c3 = (new_image_c1 + new_image_c2) / 2
        new_image = torch.cat((new_image_c1, new_image_c2, new_image_c3), 0)
        new_image = new_image.expand(1, 3, h, w)
        return new_image

    input_switcher = {
        "depth": depth_input,
        "fused": fused_input,
    }
    depth_image = normalize(depth_image, img_shape=img_shape)
    ir_image = normalize(ir_image, img_shape=img_shape)

    return input_switcher[input_type](depth_image=depth_image, ir_image=ir_image)


#########################
##### Model Helpers #####
#########################

def find_prediction_mask(pred_heatmap: torch.tensor, window_size=11, threshold=const.THRESHOLD_ACC):
    """
    Find the mask of a giver heatmap, Have this in mind the follwoing heatmap might not have values as larg as
    1, and we need to fins the local maximas of the heatmap.

    :param pred_heatmap: torch.tensor, predicted heatmap by the model
    :param window_size: int, size of the maxPooling window
    :return: torch.tensor (mask of the heatmap)
    """
    pred_local_max = torch.max_pool2d(pred_heatmap[None, None, ...], kernel_size=window_size, stride=1, padding=window_size//2)
    return (pred_local_max == pred_heatmap) * (pred_heatmap > threshold)

def get_bboxes(yx_locations: torch.tensor, height: torch.tensor, width: torch.tensor,\
        offset_x: torch.tensor, offset_y: torch.tensor, stride=const.CENTERNET_STRIDE, img_shape=const.CENTERNET_IMG_SHAPE):
    """
    Create a list of bounding boxes [[xmin, ymin, xmax, ymax], ...]

    :param yx_locations: torch.tensor, X and Y locations in the heatmap has to be mutiplied by the stride to go back to original dims
    :param height: torch.tensor, The height of the bbox 
    :param width: torch.tensor, The width of the bbox
    :param offset_x: torch.tensor, The X offset value
    :param offst_y: torch.tensor, The Y offset value
    """
    yx_locations *= stride
    bboxes = []
    for i, yx_location in enumerate(yx_locations):
        y_center = yx_location[0].item() + offset_y[i].item()
        x_center = yx_location[1].item() + offset_x[i].item()
        h = height[i].item()
        w = width[i].item()

        x_min = max(0, x_center - w/2)
        y_min = max(0, y_center - h/2)

        bboxes.append([x_min, y_min, w, h])
    
    return bboxes

def get_median_depth(img, xy_locs:list):
    """
    Get the median depth of the hand

    :param img: numpy array, depth image
    :param xy_locs: list, [x_min, y_min, x_max, y_max] locations of the bounding box
    :return: float, median depth
    """
    return np.median(img)

def a2j_depth_image_transform(img, xy_locs: list, target_size=const.A2J_TARGET_SIZE, depth_thresh=const.DEPTH_THRESHOLD):
    """
    Transform the depth image to appropriate format for running through the model

    :param img: numpy array, depth image
    :param xy_locs: list, [x_min, y_min, x_max, y_max] locations of the bounding box
    :param target_size: tuple, input target size of the A2J network
    :paran depth_thresh: int, depth threshold to 0 out the unwanted pixels
    :return: processed depth image to feed into the a2j
    """

    img_output = np.ones((target_size[1], target_size[0], 1), dtype="float32")

    new_Xmin = xy_locs[0]
    new_Ymin = xy_locs[1]
    new_Xmax = xy_locs[2]
    new_Ymax = xy_locs[3]

    img_crop = img[new_Ymin:new_Ymax, new_Xmin:new_Xmax]
    median_depth = get_median_depth(img_crop, xy_locs)

    center_x = (new_Xmax+new_Xmin)/2
    center_y = (new_Ymax+new_Ymin)/2
    new_Xmin = int(max(center_x-110, 0))
    new_Ymin = int(max(center_y-110, 0))
    new_Xmax = int(min(center_x+110, img.shape[1]-1))
    new_Ymax = int(min(center_y+110, img.shape[0]-1))
    img_crop = img[new_Ymin:new_Ymax, new_Xmin:new_Xmax]
    
    img_resize = cv2.resize(img_crop, target_size, interpolation=cv2.INTER_NEAREST)
    img_resize - np.asarray(img_resize, dtype="float32")
    img_resize[np.where(img_resize >= median_depth + depth_thresh)] = median_depth 
    img_resize[np.where(img_resize <= median_depth - depth_thresh)] = median_depth
    img_resize = (img_resize - median_depth)
    img_resize = (img_resize - MEAN)/STD
    
    img_output[:,:,0] = img_resize


    img_output = np.asarray(img_output)
    img_NCHW_out = img_output.transpose(2, 0, 1)
    img_NCHW_out = np.asarray(img_NCHW_out)

    img_out = torch.from_numpy(img_NCHW_out)
    img_out = img_out.unsqueeze(0)

    # n, c, h, w = img_out.size()
    # img_out = img_out.expand(n, 3, h, w)

    return img_out, median_depth

def back_to_normal(pred_joints, xy_locs:list, median_depths:float, target_size=const.A2J_TARGET_SIZE):
    """
    Transform the predicted joint to the original space

    :param pred_joints: list of np.array, list of predicted joints
    :param xy_locs: list, [x_min, y_min, x_max, y_max] locations of the bounding box
    :param median_depth: float, the value of median depth
    """

    normalized_joints = []
    for i in range(len(pred_joints)):
        pred_joint = pred_joints[i].cpu()
        pred_joint = pred_joint.detach().numpy()

        xy_bb = xy_locs[i]
        median_depth = median_depths[i]

        p_j = np.ones((const.NUM_JOINTS, 3))
        x_len = abs(xy_bb[0] - xy_bb[2])
        y_len = abs(xy_bb[1] - xy_bb[3])

        p_j[:,0] = ((pred_joint[:,1] * x_len) / target_size[0]) + xy_bb[0]
        p_j[:,1] = ((pred_joint[:,0] * y_len) / target_size[1]) + xy_bb[1]
        p_j[:,2] = pred_joint[:,2] + median_depth

        normalized_joints.append(p_j)
    
    return normalized_joints


def get_xyz_lims(pred_joints_collections):
    max_range = [0, 0, 0]  
    min_range = [float("inf"), float("inf"), float("inf")] 

    for pred_joints in pred_joints_collections:
        min_x = pred_joints[:,0].min()
        if min_x < min_range[0]:  
            min_range[0] = min_x  
        min_y = pred_joints[:,1].min()
        if min_y < min_range[1]:  
            min_range[1] = min_y  
        min_z = pred_joints[:,2].min()
        if min_z < min_range[2]:  
            min_range[2] = min_z  

        max_x = pred_joints[:,0].max()
        if max_x > max_range[0]:  
            max_range[0] = max_x  
        max_y = pred_joints[:,1].max()
        if max_y > max_range[1]:  
            max_range[1] = max_y  
        max_z = pred_joints[:,2].max()
        if max_z > max_range[2]:  
            max_range[2] = max_z

    return max_range, min_range

def vizualize_frams(ax_2, pred_joints_collections):
    pred_joints_collections = np.array(pred_joints_collections)
    
    max_range, min_range = get_xyz_lims(pred_joints_collections)
    
    mid_x = (max_range[0] + min_range[0])/2 
    mid_y = (max_range[1] + min_range[1])/2 
    mid_z = (max_range[2] + min_range[2])/2

    # Second subplot
    ax_2.grid(True)
    ax_2.set_xticklabels([])
    ax_2.set_yticklabels([]) 
    ax_2.set_zticklabels([])

    ax_2.set_xlim(mid_x - max_range[0]/2, mid_x + max_range[0]/2) 
    ax_2.set_ylim(mid_y - max_range[1]/2, mid_y + max_range[1]/2) 
    ax_2.set_zlim(mid_z - max_range[2]/2, mid_z + max_range[2]/2) 

    scats = []
    for pred_joints in pred_joints_collections:
        ax_2.scatter(pred_joints[:,0], pred_joints[:,1], pred_joints[:,2], c='r', marker='^', s=10)
        
        # MY SCRIPT
        if const.NUM_JOINTS == 36:
            ax_2.plot(pred_joints[0:6,0], pred_joints[0:6,1], pred_joints[0:6,2], color='b')
            ax_2.plot(pred_joints[6:12,0], pred_joints[6:12,1], pred_joints[6:12,2], color='b')
            ax_2.plot(pred_joints[12:18,0], pred_joints[12:18,1], pred_joints[12:18,2], color='b')
            ax_2.plot(pred_joints[18:24,0], pred_joints[18:24,1], pred_joints[18:24,2], color='b')
            ax_2.plot(pred_joints[24:30,0], pred_joints[24:30,1], pred_joints[24:30,2], color='b')
        

        # MY SCRIPT 16 JOINTS
        if const.NUM_JOINTS == 16:
            ax_2.plot(pred_joints[0:3,0], pred_joints[0:3,1], pred_joints[0:3,2], color='b')
            ax_2.plot(pred_joints[3:6,0], pred_joints[3:6,1], pred_joints[3:6,2], color='b')
            ax_2.plot(pred_joints[6:9,0], pred_joints[6:9,1], pred_joints[6:9,2], color='b')
            ax_2.plot(pred_joints[9:12,0], pred_joints[9:12,1], pred_joints[9:12,2], color='b')
            ax_2.plot(pred_joints[12:15,0], pred_joints[12:15,1], pred_joints[12:15,2], color='b')
            ax_2.plot([pred_joints[2,0], pred_joints[15,0]], [pred_joints[2,1], pred_joints[15,1]], [pred_joints[2,2], pred_joints[15,2]], color='b')
            ax_2.plot([pred_joints[5,0], pred_joints[15,0]], [pred_joints[5,1], pred_joints[15,1]], [pred_joints[5,2], pred_joints[15,2]], color='b')
            ax_2.plot([pred_joints[8,0], pred_joints[15,0]], [pred_joints[8,1], pred_joints[15,1]], [pred_joints[8,2], pred_joints[15,2]], color='b')
            ax_2.plot([pred_joints[11,0], pred_joints[15,0]], [pred_joints[11,1], pred_joints[15,1]], [pred_joints[11,2], pred_joints[15,2]], color='b')
            ax_2.plot([pred_joints[14,0], pred_joints[15,0]], [pred_joints[14,1], pred_joints[15,1]], [pred_joints[14,2], pred_joints[15,2]], color='b')
        


    ax_2.view_init(-70, -70) 

    return scats
