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
        confidence_percentage = "{0:.0%}".format(confidence)
        print("Detected {} with confidence {}".format(
            class_name, confidence_percentage))
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
    parser.add_argument('-i', '--input_img_path',  type=str, default='dog.jpg',
        help='an image file to run inference on')
    parser.add_argument('-p', '--precision', type=int, choices=[32, 16], default=32,
        help='desired TensorRT float precision to build an engine with')
    parser.add_argument('-b', '--max_batch_size', type=int, default=1,
        help='max TensorRT engine batch size')
    parser.add_argument('-w', '--workspace_dir',
        help='sample workspace directory')
    parser.add_argument("-o", "--output",
        help="path of the output file",
        default="image_inferred.jpg")

    # Parse arguments passed
    args = parser.parse_args()


    # Fetch TensorRT engine path and datatype
    args.trt_engine_datatype = TRT_PRECISION_TO_DATATYPE[args.precision]
    args.trt_engine_path = "ssd_inception_v2_trt_engine.buf"
    
    return args

def main():
    # Parse command line arguments
    args = parse_commandline_arguments()

    # Set up all TensorRT data structures needed for inference
    trt_inference_wrapper = inference_utils.TRTInference(
        args.trt_engine_path, 
        trt_engine_datatype=args.trt_engine_datatype,
        batch_size=args.max_batch_size)
    cv_image = cv2.imread(args.input_img_path)
    detection_out, keep_count_out = \
        trt_inference_wrapper.infer(cv_image)
    # Start measuring time
    inference_start_time = time.time()

    # Get TensorRT SSD model output
    detection_out, keep_count_out = \
        trt_inference_wrapper.infer(cv_image)

    # Make PIL.Image for drawing bounding boxes and
    # let analyze_prediction() draw them based on model output
    prediction_fields = len(TRT_PREDICTION_LAYOUT)
    for det in range(int(keep_count_out[0])):
        analyze_prediction(detection_out, det * prediction_fields, cv_image)

    # Output total [img load + inference + drawing bboxes] time
    print("Total time taken for one image: {} ms\n".format(
        int(round((time.time() - inference_start_time) * 1000))))

    # Save output image and output path
    cv2.imshow("test", cv_image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imwrite(args.output, cv_image)
    print("Saved output image to: {}".format(args.output))


if __name__ == '__main__':
    main()