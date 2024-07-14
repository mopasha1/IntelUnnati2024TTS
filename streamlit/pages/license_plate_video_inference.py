import streamlit as st
from ultralytics import YOLO
import cv2
from model.LPRNET import LPRNet, CHARS
from model.STN import STNet
import torch
import cv2
import sys
import numpy as np
import time
import pandas as pd
import warnings
import tempfile

warnings.filterwarnings("ignore")




sys.path.append("../src/")
st.set_page_config(layout="wide")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = YOLO(r"../models/license_plate_detection_models/best_120_epoch_int8_openvino_model_640", task="detect")
lprnet = LPRNet(class_num=len(CHARS), dropout_rate=0)
lprnet.to(device)
lprnet.load_state_dict(
    torch.load(
        "./weights/training_checkpoint_files/lprnet_Iter_015300_model.ckpt",
        map_location=lambda storage, loc: storage,
    )["net_state_dict"]
)
lprnet.eval()
# 4 sec -> 77 ms
STN = STNet()
STN.to(device)
STN.load_state_dict(
    torch.load(
        "./weights/training_checkpoint_files/stn_Iter_015300_model.ckpt",
        map_location=lambda storage, loc: storage,
    )["net_state_dict"]
)
STN.eval()
print("Successfully built the network!")


# Function to read the uploaded video file using OpenCV
def read_video(file):
    # Create a temporary file to store the uploaded video
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(file.read())
    temp_file.close()

    # Open the video file with OpenCV
    cap = cv2.VideoCapture(temp_file.name)

    return cap, temp_file.name



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
    im = cv2.resize(roi, roi.shape[:2][::-1])  # fixing the rounding error

    if (bbox_height) / (bbox_width) > 0.4:
        width, height = im.shape[:2]
        half = width // 2
        curr1 = im[:half, : height - height // 20]
        curr2 = im[half:, height // 20 :]
        # print(curr1.shape, curr2.shape  )
        if curr1.shape[0] < curr2.shape[0]:
            curr1 = im[: half + 1, : height - height // 20]
        stacked = np.hstack((curr1, curr2))
        im = cv2.resize(stacked, (94, 24))
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # cv2.imshow('testing', im)
        # cv2.waitKey(0)
    else:
        im = cv2.resize(im, (94, 24))
    # cv2.imshow('testing', im)
    # cv2.waitKey(0)
    im = (np.transpose(np.float32(im), (2, 0, 1)) - 127.5) * 0.0078125  # Normalization

    data = (
        torch.from_numpy(im).float().unsqueeze(0).to(device)
    )  # torch.Size([1, 3, 24, 94])
    transfer = STN(data)
    preds = lprnet(transfer)
    preds = preds.cpu().detach().numpy()  # (1, 37, 18)

    labels, pred_labels = decode(preds, CHARS)
    return labels[0].lstrip("-").rstrip("-")


def video_inference(video, video_path, model, frame_log_container, output_container):
    # Process each frame using the generator
    # Process the image
    sum_lpr_time = 0
    sum_yolo_time = 0
    count = 0
    cap = video
    # Get video properties
    # Create VideoWriter object to save the output video
    # Process results generator
    for result in model(video_path, stream=True):
        # Get bounding boxes from YOLOv8 results
        yolo_time = result.speed.get("inference")
        sum_yolo_time += yolo_time
        bboxes = result.boxes
        # print(bboxes.data)
        # Read frame from the video
        data = pd.DataFrame({"frames": 0, "inference speed": 0.0}, index=[0])
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
            # Example: Pass the ROI to another model (dummy processing here)
            lpr_time = time.perf_counter()
            class_label = process_roi(roi, bbox_width, bbox_height)
            lpr_end_time = time.perf_counter()
            sum_lpr_time += lpr_end_time - lpr_time
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
            output_container.image(frame, channels="BGR")
            frame_log_container.subheader(
                f"Frame {count+1}: Detected Plate: {class_label if class_label else ''} LPRNet Inference Time: {round((lpr_end_time - lpr_time)*1000,3)} ms, YOLO Inference Time: {round(yolo_time,3)} ms\n"
            )
            count += 1
    cap.release()
    # Save the logs
    return count, sum_lpr_time / count, sum_yolo_time / count

# Streamlit app

st.header("Upload a video file")
video = st.file_uploader("Upload a file", type=["mp4", "avi", "mov", "mkv"])
if video is not None:
    # Convert the uploaded video to numpy array
    st.session_state["video_loaded"] = True
    uploaded_video, video_path = read_video(video)
    print(video_path)
    # Pass the video array to the video_inference function
    try:
        video_columns = st.columns(2, gap="large")
        # Display the output video
        with video_columns[0]:
            st.subheader("Original Video")
            st.video(video)
        with video_columns[1]:
            st.subheader("Detections")
            empty = st.empty()
            log_container = st.empty()
        chart_container = st.empty()
        count, avg_lpr_time, avg_yolo_time = video_inference(
            uploaded_video, video_path, model, log_container, empty
        )
        columns = st.columns(3)

        with columns[0]:
            st.metric("Total Frames in Video", count)

        with columns[1]:
            st.metric(
                "Average LPRNet Inference Time (ms)",
                round(avg_lpr_time * 1000, 3),
            )

        with columns[2]:
            st.metric("Average YOLO Inference Time (ms)", round(avg_yolo_time, 3))
        st.session_state["video_loaded"] = False
    except Exception as e:
        print(e)
        st.error(
            "An error occurred during inference: No license plates detected. Please try again with a different video file."
        )
        st.session_state["video_loaded"] = False
else:
    st.write("Please upload an video file.")
