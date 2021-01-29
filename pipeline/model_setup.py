'''
Copyright (c) 2019 Boshen Zhang
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import os
import torch

# PROJ LIBRARY
import pipeline.constants as const

from model.CenterNet.centernet import Resnet18FeatureExtractor

from model.A2J.a2j import A2J
from model.A2J.model import A2J_model
from model.A2J.a2j_utilities.post_processing import PostProcess


class ModelSetup(object):
    """
    Class to setup Both SSD and A2J model
    """
    def __init__(self, BBOX_MODEL_PATH:str, A2J_model_path=const.A2J_MODEL_PATH, trt_optim=False):
        """
        
        :param SSD_model_path: string, full path to ssd Model checkpoint
        :param A2J_model_path: string, full path to A2J Model checkpoint
        """
        self.bb_model_path = BBOX_MODEL_PATH
        self.a2j_path = A2J_model_path
    
        print("Loading CenterNet ...")
        centernet_check_point = torch.load(self.bb_model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.centernet_model_name = centernet_check_point["model_name"]
        self.centernet_num_classes = centernet_check_point['num_classes']
        self.bb_model = Resnet18FeatureExtractor(num_classes=self.centernet_num_classes)
        self.bb_model.load_state_dict(centernet_check_point["model"])
        print("CenterNet Loading Finished! 

")

        if trt_optim:
            import tensorrt as trt
            from torch2trt import torch2trt, TRTModule
            trt_model_path = self.bb_model_path.split(".")[0] + ".trt"
            if not os.path.exists(trt_model_path):
                print("Creating TRT Bounding Box Model...")
                i
                x = torch.ones((1, 3, const.INPUT_IMG_SIZE[0], const.INPUT_IMG_SIZE[1])).cuda()

                self.bb_model = torch2trt(self.bb_model.eval().cuda(), [x], fp16_mode=True)
                torch.save(self.bb_model.state_dict(), trt_model_path)
                print(f"TRT Bounding Box Model saved at:
 {trt_model_path}
")
    
            print("Loading TRT Bounding Box Model...")
            del self.bb_model

            self.bb_model = TRTModule()
            self.bb_model.load_state_dict(torch.load(trt_model_path))  
            print("TRT Bounding Box Model loaded!
")

        # Load A2J model
        print("Loading A2J ...")
        backbone_name = [elem[0] for idx, elem in enumerate(const.A2J_BACKBONE_NAME.items()) if elem[1]][0]
        a2j_check_point = torch.load(self.a2j_path, map_location=torch.device("cpu"))

        self.num_class = a2j_check_point["num_classes"]
        # self.a2j_model = A2J_model(num_classes=self.num_class)
        self.a2j_model = A2J(num_joints=self.num_class, backbone_name=backbone_name, backbone_pretrained=True)
        self.a2j_model.load_state_dict(a2j_check_point["model"])
        self.post_process =  PostProcess(shape=(const.A2J_TARGET_SIZE[1]//16, const.A2J_TARGET_SIZE[0]//16),\
                                            stride=const.A2J_STRIDE)

        if trt_optim:
            from torch2trt import torch2trt, TRTModule
            trt_a2j_model_path = self.a2j_path.split(".")[0] + ".trt"
            if not os.path.exists(trt_a2j_model_path):
                print("Creating TRT A2J Model...")
                x = torch.empty((1, 1, const.A2J_TARGET_SIZE[0], const.A2J_TARGET_SIZE[1])).cuda().float()

                self.a2j_model = torch2trt(self.a2j_model.eval().cuda(), [x], fp16_mode=True)
                torch.save(self.a2j_model.state_dict(), trt_a2j_model_path)
                print(f"TRT A2J Model saved at:
 {trt_a2j_model_path}
")
            
            print("Loading TRT A2J Model...")
            del self.a2j_model
            
            self.a2j_model = TRTModule()
            self.a2j_model.load_state_dict(torch.load(trt_a2j_model_path))  
            print("TRT A2J Model loaded!
")

        print("A2J Loading Finished! 

")
