from ultralytics import YOLO
import cv2
from model.LPRNET import LPRNet, CHARS
from model.STN import STNet
import argparse
import torch
import cv2
import numpy as np
from datetime import datetime
import pandas as pd
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.getcwd())  # adding current directory to path


def decode(preds, CHARS):
    """
    This function decodes the labels from the predictions tensor
    """
    pred_labels, labels = [], []
    n = preds.shape[0]
    
    for i in range(n):
        pred = preds[i, :, :]
        m = pred.shape[1]
        pred_label = []

        # using the maximum probability character as the predicted character

        for j in range(m):
            pred_label.append(np.argmax(pred[:, j], axis=0))
        non_repeated = [pred_label[0]]
        prev = pred_label[0]
        # Dropping repeated characters
        for c in pred_label[1:]:
            if (prev == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    prev = c
                continue
            non_repeated.append(c)
            prev = c
        pred_labels.append(non_repeated)

    for i, label in enumerate(pred_labels):
        lb = ""
        for i in label:
            lb += CHARS[i]
        labels.append(lb)

    return labels, np.array(pred_labels)


def process_roi(roi, bbox_width, bbox_height):
    """
    Using the STN and LPRNet models to process the region of interest (ROI) that is detected from the YOLOv8 model.
    Returns the predicted label for the ROI
    """
    im = cv2.resize(roi, roi.shape[:2][::-1])  # fixing the rounding error
    # cv2.imshow('testing', im)
    # cv2.waitKey(0)
    if (bbox_height) / (bbox_width) > 0.4:  # Two line license plate
        width, height = im.shape[:2]
        half = width // 2
        curr1 = im[:half, : height - height // 20]
        curr2 = im[half:, height // 20 :]
        # print(curr1.shape, curr2.shape  )
        if curr1.shape[0] < curr2.shape[0]:
            curr1 = im[: half + 1, : height - height // 20]
        stacked = np.hstack((curr1, curr2))
        im = cv2.resize(stacked, (94, 24))

    else:
        im = cv2.resize(im, (94, 24))
    im = (np.transpose(np.float32(im), (2, 0, 1)) - 127.5) * 0.0078125  # Normalization
    data = (
        torch.from_numpy(im).float().unsqueeze(0).to(device)
    )  # torch.Size([1, 3, 24, 94]) for the STN models
    transfer = STN(data)
    preds = lprnet(transfer)
    preds = (
        preds.cpu().detach().numpy()
    )  # (1, 37, 18) is the shape of the output tensor for LPRNET

    labels, _ = decode(preds, CHARS)
    return labels[0]


def video_inference(video_path, output_path, model, logs):
    """
    Function for processing a video file frame by frame and extracting license plates from it.
    """
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    # Create VideoWriter object to save the output video
    # Process results generator
    for result in model(video_path, stream=True):
        # Get bounding boxes from YOLOv8 results
        bboxes = result.boxes
        # print(bboxes.data)
        # Read frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Process each bounding box
        for bbox in bboxes.data:
            x1, y1, x2, y2, conf, cls = (
                int(bbox[0]),
                int(bbox[1]),
                int(bbox[2]),
                int(bbox[3]),
                float(bbox[4]),
                int(bbox[5]),
            )

            # Extract the region of interest (ROI) using the bounding box coordinates
            roi = frame[y1:y2, x1:x2]
            bbox_width = x2 - x1
            bbox_height = y2 - y1

            # Pass the ROI to LPRNet model

            class_label = process_roi(roi, bbox_width, bbox_height)
            current_time = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )  # Format: YYYY-MM-DD HH:MM:SS.mmm
            # logging the output to logs
            if len(logs) == 0 or len(logs[logs["License Plate"] == class_label]) == 0:
                logs = pd.concat(
                    [
                        logs,
                        pd.DataFrame(
                            {
                                "Time": current_time,
                                "License Plate": class_label,
                                "Confidence": conf,
                                "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2,
                            },
                            index=[0],
                        ),
                    ],
                    ignore_index=True,
                )
            text_width, text_height = cv2.getTextSize(
                class_label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2
            )[0]
            cv2.rectangle(frame, (x1, y1 - text_height * 2), (x2, y1), (0, 0, 0), -1)
            cv2.putText(
                frame,
                f"{class_label}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Write the frame to the output video
        out.write(frame)

        # Optional: Display the frame (uncomment to display)
        if args["show"]:
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    logs.to_csv("logs.csv", index=False)
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(
        "Output video saved at: ",
        args["output"] if args["output"] else "output_video.mp4",
    )


def image_inference(image_path, output_image_path, model, logs):
    """
    Function to process a single image and extract license plates from it.
    """
    image = cv2.imread(image_path)
    # Process the image
    for result in model(image):
        # print(result)
        bboxes = result.boxes

        # Process each bounding box
        for bbox in bboxes.data:
            x1, y1, x2, y2, conf, cls = (
                int(bbox[0]),
                int(bbox[1]),
                int(bbox[2]),
                int(bbox[3]),
                float(bbox[4]),
                int(bbox[5]),
            )

            # Extract the region of interest (ROI) using the bounding box coordinates
            roi = image[y1:y2, x1:x2]
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            # Pass the ROI to LPRNet model
            class_label = process_roi(roi, bbox_width, bbox_height)
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[
                :-3
            ]  # Format: YYYY-MM-DD HH:MM:SS.mmm
            if len(logs) == 0 or len(logs[logs["License Plate"] == class_label]) == 0:
                logs = pd.concat(
                    [
                        logs,
                        pd.DataFrame(
                            {
                                "Time": current_time,
                                "License Plate": class_label,
                                "Confidence": conf,
                                "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2,
                            },
                            index=[0],
                        ),
                    ],
                    ignore_index=True,
                )
            # Display the class result of the other model
            # Calculate the width and height of the text

            text_width, text_height = cv2.getTextSize(
                class_label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2
            )[0]
            cv2.rectangle(image, (x1, y1 - text_height * 4), (x2, y1), (0, 0, 0), -1)
            cv2.putText(
                image,
                f"{class_label}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                2,
            )
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            print("Detected License Plate:", class_label)

    if args["show"]:
        cv2.imshow("annotated", cv2.resize(image, (800, 600)))
        cv2.waitKey(0)
    # Save the logs
    cv2.imwrite(output_image_path, image)
    print("Output image saved at:", output_image_path)
    logs.to_csv("logs.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--image", help="path to input image")
    group.add_argument("-v", "--video", help="path to input video")
    ap.add_argument(
        "-show", "--show", default=False, help="show output annotated image"
    )
    ap.add_argument(
        "--output", "-o", default=None, help="path to output annotated image or video"
    )
    args = vars(ap.parse_args())

    logs = pd.DataFrame(
        columns=["Time", "License Plate", "Confidence", "x1", "y1", "x2", "y2"]
    )
    # Load YOLOv8 model
    model = YOLO(r"../models/license_plate_detection_models/best_120_epoch_int8_openvino_model_640", task="detect")
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )  # Device agnostic

    lprnet = LPRNet(class_num=len(CHARS), dropout_rate=0)
    lprnet.to(device)
    # lprnet.load_state_dict(torch.load('./weights/training_checkpoint_files/lprnet_Iter_015300_model.ckpt', map_location=lambda storage, loc: storage)['net_state_dict'])
    lprnet.load_state_dict(
        torch.load(
            r"../models/license_plate_ocr_models/lprnet_Iter_043200_model.ckpt",
            map_location=lambda storage, loc: storage,
        )["net_state_dict"]
    )

    lprnet.eval()
    # 4 sec -> 77 ms
    STN = STNet()
    STN.to(device)
    # STN.load_state_dict(torch.load('./weights/training_checkpoint_files/stn_Iter_015300_model.ckpt', map_location=lambda storage, loc: storage)['net_state_dict'])
    STN.load_state_dict(
        torch.load(
            r"../models/license_plate_ocr_models/stn_Iter_043200_model.ckpt",
            map_location=lambda storage, loc: storage,
        )["net_state_dict"]
    )

    STN.eval()
    print("Successfully built the network!")

    video_path = None
    image_path = None

    # Open video file
    if args["video"]:
        video_path = args["video"]
    else:
        image_path = args["image"]

    if video_path:
        if args["output"]:
            out = args["output"]
        else:
            out = "output_video.mp4"
        video_inference(video_path, out, model, logs)

    # Open image file
    elif image_path:
        if args["output"]:
            out = args["output"]
        else:
            out = "output_image.jpg"
        image_inference(image_path, out, model, logs)
