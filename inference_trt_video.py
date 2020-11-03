import os
import ctypes
import time
import sys
import argparse

import numpy as np
import cv2
import tensorrt as trt
import coco as coco_utils
import boxes as boxes_utils
import inference as inference_utils # TRT/TF inference wrappers

COCO_LABELS = coco_utils.COCO_CLASSES_LIST
MODEL_NAME = 'ssd_inception_v2_trt_engine.buf'
VISUALIZATION_THRESHOLD = 0.4
TRT_PRECISION_TO_DATATYPE = {
    16: trt.DataType.HALF,
    32: trt.DataType.FLOAT
}
TRT_PREDICTION_LAYOUT = {
    "image_id": 0,
    "label": 1,
    "confidence": 2,
    "xmin": 3,
    "ymin": 4,
    "xmax": 5,
    "ymax": 6
}

def fetch_prediction_field(field_name, detection_out, pred_start_idx):
    """Fetches prediction field from prediction byte array.

    After TensorRT inference, prediction data is saved in
    byte array and returned by object detection network.
    This byte array contains several pieces of data about
    prediction - we call one such piece a prediction field.
    The prediction fields layout is described in TRT_PREDICTION_LAYOUT.

    This function, given prediction byte array returned by network,
    staring index of given prediction and field name of interest,
    returns prediction field data corresponding to given arguments.

    Args:
        field_name (str): field of interest, one of keys of TRT_PREDICTION_LAYOUT
        detection_out (array): object detection network output
        pred_start_idx (int): start index of prediction of interest in detection_out

    Returns:
        Prediction field corresponding to given data.
    """
    return detection_out[pred_start_idx + TRT_PREDICTION_LAYOUT[field_name]]

def analyze_prediction(detection_out, pred_start_idx, img_cv):
    image_id = int(fetch_prediction_field("image_id", detection_out, pred_start_idx))
    label = int(fetch_prediction_field("label", detection_out, pred_start_idx))
    confidence = fetch_prediction_field("confidence", detection_out, pred_start_idx)
    xmin = fetch_prediction_field("xmin", detection_out, pred_start_idx)
    ymin = fetch_prediction_field("ymin", detection_out, pred_start_idx)
    xmax = fetch_prediction_field("xmax", detection_out, pred_start_idx)
    ymax = fetch_prediction_field("ymax", detection_out, pred_start_idx)
    if confidence > VISUALIZATION_THRESHOLD:
        class_name = COCO_LABELS[label]
        if class_name == "person":
            if (xmax - xmin) > 0.5:
                print("warning! person is too close!")
        confidence_percentage = "{0:.0%}".format(confidence)
        boxes_utils.draw_bounding_boxes_on_image(
            img_cv, np.array([[ymin, xmin, ymax, xmax]]),
            display_str_list=["{}: {}".format(
                class_name, confidence_percentage)],
            color=coco_utils.COCO_COLORS[label]
        )

def parse_commandline_arguments():
    """Parses command line arguments and adjusts internal data structures."""

    # Define script command line arguments
    parser = argparse.ArgumentParser(description='Run object detection inference on input image.')
    parser.add_argument('-i', '--input_video',  type=str, default='v4l2src device=/dev/video1 ! video/x-raw, width=640, height=360, format=(string)YUY2,framerate=30/1 ! videoconvert ! video/x-raw,width=640,height=360,format=BGR ! appsink',
    #parser.add_argument('-i', '--input_video',  type=str, default='nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)360, format=(string)NV12, framerate=(fraction)60/1 ! nvvidconv flip-method=0 ! video/x-raw, width=(int)640, height=(int)360, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink',
    #parser.add_argument('-i', '--input_video',  type=str, default='challenge.mp4',
        help='an image file to run inference on')
    parser.add_argument('-p', '--precision', type=int, choices=[32, 16], default=32,
        help='desired TensorRT float precision to build an engine with')
    parser.add_argument('-b', '--max_batch_size', type=int, default=1,
        help='max TensorRT engine batch size')
    parser.add_argument('-w', '--workspace_dir',
        help='sample workspace directory')
    parser.add_argument("-o", "--output",
        help="path of the output file",
        default="result.avi")

    # Parse arguments passed
    args = parser.parse_args()


    # Fetch TensorRT engine path and datatype
    args.trt_engine_datatype = TRT_PRECISION_TO_DATATYPE[args.precision]
    args.trt_engine_path = "ssd_inception_v2_trt_engine.buf"
    
    return args

def Video(savepath = None):
    args = parse_commandline_arguments()
    cap = cv2.VideoCapture(args.input_video)
    if cap.isOpened():
        print("Video Opened")
    else:
        print("Video Not Opened")
        print("Program Abort")
        exit()
    trt_inference_wrapper = inference_utils.TRTInference(
        args.trt_engine_path, 
        trt_engine_datatype=args.trt_engine_datatype,
        batch_size=args.max_batch_size)
    _, _ = trt_inference_wrapper.infer(np.random.rand(300, 300, 3))
    prediction_fields = len(TRT_PREDICTION_LAYOUT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = 640
    height = 360
    out = None
    if savepath is not None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(savepath, fourcc, fps, (640, 360), True)
    cv2.namedWindow("Frame", cv2.WINDOW_GUI_EXPANDED)
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Our operations on the frame come here
            inference_start_time = time.time()
            detection_out, keep_count_out = \
                trt_inference_wrapper.infer(frame)
            
            for det in range(int(keep_count_out[0])):
                analyze_prediction(detection_out, det * prediction_fields, frame)
            print("Total time taken for one frame: {} ms\n".format(
                int(round((time.time() - inference_start_time) * 1000))))
            if out is not None:
                # Write frame-by-frame
                out.write(frame)
            # Display the resulting frame
            cv2.imshow("Frame", frame)
        else:
            break
        # waitKey(int(1000.0/fps)) for matching fps of video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    return

if __name__ == '__main__':
    Video()
