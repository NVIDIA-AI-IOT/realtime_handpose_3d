# 3D_HandPose

This repository implements a realtime 3D hand posture estimation pipeline running on Jetson platform using a [**Azure Kinect** camera](https://azure.microsoft.com/en-us/services/kinect-dk/).<br/>
Please refer to the following repositories before getting started here:
- [centernet_kinet](https://github.com/NVIDIA-AI-IOT/centernet_kinect)
- [Hand Posture Estimation](https://github.com/NVIDIA-AI-IOT/a2j_handpose_3d)

<p align="center">
<img src="readme_files/realtime_inference.gif" alt="landing graphic" height="600px"/>
</p>


There are 2 stages to our pipeline

* ## [CenterNet Bounding Box](#centernet_bounding_box)
* ## [A2J Posture Detection](#a2j_posture_detection)
* ## [Run inference](#run_infrence)

<a name="centernet_bounding_box"></a>
## CenterNet Bounding Box

The first stage will localize the hand using a fusion of infrared and depth image.<br/>
**NOTE:** more detail can be found in the centernet_kinect repository

<a name="a2j_posture_detection"></a>
## A2J Posture Detection

The second stage would perform 3D hand posture estimation on the region of intrest selected by the previous step.<br/>
**NOTE:** for training a model please refer to the Hand Posture Estimation repository

<a name="run_infrence"></a>
## Run inference

- Initially configure the *pipeline/constants.py* file:
  - **CENTERNET_MODEL_PATH** please place the centernet model weights in *"/checkpoint/CenterNet"*<br/>
  with the naming convention that was provided in the original repository
    - Configure the centernet portion of the file as its been described in the original [repository](https://github.com/NVIDIA-AI-IOT/centernet_kinect#get_pre_trained_weights).<br/>
    if you are using the weights directly from the original repository you dont have to modify this section.
  - **A2J_MODEL_PATH** please place the A2J model weights in *"/checkpoint/A2J"*<br/>
  with the naming convention that was provided in the original repository
    - Configure the a2j portion of the file as you have set up the training pipeline for [Hand Posture Estimation](https://github.com/NVIDIA-AI-IOT/a2j_handpose_3d).<br/>
    - For Faster inference we use TensorRT inference engine to optimize the models. this will take some time to compile the models and create a TRT engine<br/>
- Run realtime inference on a jetson platform.
    ```bash
    cd pipeline
    python3 azure_kinect.py
    
    # Optional for faster inference
    python3 azure_kinect.py --trt True # for optimizing the models with TensorRT fp16  
    ```
