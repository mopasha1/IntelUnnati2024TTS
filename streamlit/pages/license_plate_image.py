import streamlit as st
from ultralytics import YOLO
import cv2
from model.LPRNET import LPRNet, CHARS
from model.STN import STNet
import torch
import cv2
import sys
import numpy as np
from PIL import Image
import pandas as pd
import time

sys.path.append("../src/")
st.set_page_config(layout="wide")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = YOLO(r"../models/license_plate_detection_models/best_120_epoch_int8_openvino_model_640", task="detect")
lprnet = LPRNet(class_num=len(CHARS), dropout_rate=0)
lprnet.to(device)
lprnet.load_state_dict(
    torch.load(
        "./weights/lprnet_Iter_043200_model.ckpt",
        map_location=lambda storage, loc: storage,
    )["net_state_dict"]
)
lprnet.eval()
# 4 sec -> 77 ms
STN = STNet()
STN.to(device)
STN.load_state_dict(
    torch.load(
        "./weights/stn_Iter_043200_model.ckpt",
        map_location=lambda storage, loc: storage,
    )["net_state_dict"]
)
STN.eval()
print("Successfully built the network!")


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


def image_inference(image, model):
    # Process the image
    sum_lpr_time = 0
    sum_yolo_time = 0
    count = 0
    rois = []
    class_labels = []
    confidences = []
    for result in model(image):
        yolo_time = result.speed.get("inference")
        sum_yolo_time += yolo_time
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
            roi = image[y1:y2, x1:x2, :]
            print(roi.shape)
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            # Example: Pass the ROI to another model (dummy processing here)
            lpr_time = time.perf_counter()
            class_label = process_roi(roi, bbox_width, bbox_height)
            lpr_end_time = time.perf_counter()
            sum_lpr_time += lpr_end_time - lpr_time
            # Display the class result of the other model
            # Calculate the width and height of the text

            text_width, text_height = cv2.getTextSize(
                class_label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2
            )[0]
            cv2.rectangle(image, (x1, y1 - text_height * 2), (x2, y1), (0, 0, 0), -1)
            cv2.putText(
                image,
                f"{class_label}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            rois.append(roi)
            class_labels.append(class_label)
            confidences.append(conf)
            print("Detected License Plate:", class_label)
            count += 1
    # Save the logs
    return (
        image,
        rois,
        class_labels,
        confidences,
        sum_lpr_time / count,
        sum_yolo_time / count,
    )


st.header("License plate detection model")
df = pd.DataFrame({"Authorized": ["TN87E2455", "TS08FT5175", "KA53MC7074", "TS09EM4884"]})
st.subheader("List of Authorized License Plates")
if "df" not in st.session_state:
    st.session_state.df = df

input_text = st.text_input("Enter a license plate number to add to the list:")
if st.button("Add License Plate"):
    st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame({"Authorized": input_text}, index=[len(st.session_state.df)])])
    st.write("License plate added successfully!")
st.dataframe(st.session_state.df.transpose(), use_container_width=True)
st.subheader("Upload an image file")

image = st.file_uploader("Upload a file", type=["png", "webp", "jpg", "jpeg"])
if image is not None:
    # Convert the uploaded image to numpy array
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    img_array = cv2.imdecode(file_bytes, 1)

    # Pass the image array to the image_inference function
    try:
        output_image, rois, class_labels, confidences, avg_lpr_time, avg_yolo_time = (
            image_inference(img_array, model)
        )
    except:
        st.error(
            "An error occurred during inference: No license plates detected. Please try again with a different image file."
        )
    else:
        image_columns = st.columns(2, gap="large")
        # Display the output image
        with image_columns[0]:
            st.image(
                output_image,
                caption="Processed Image",
                use_column_width=True,
                channels="BGR",
            )
        with image_columns[1]:
            st.header("Detected License Plates:")
            columns = st.columns(3)

            with columns[0]:
                st.metric("Total License Plates Detected", len(class_labels))

                st.divider()
                st.subheader("Detected License Plates")
                st.write("\n\n")
            with columns[1]:
                st.metric(
                    "Average LPRNet Inference Time (ms)",
                    round(avg_lpr_time * 1000, 3),
                )

                st.divider()
                st.subheader("Region of Interest (ROI)")
                st.write("\n\n")
            with columns[2]:
                st.metric("Average YOLO Inference Time (ms)", round(avg_yolo_time, 3))

                st.divider()
                st.subheader("Confidence Scores")
                st.write("\n\n")

            with columns[0]:
                for label in class_labels:
                    st.header("\n\n")
                    st.header(label)
                    st.header("\n\n")

            with columns[1]:
                for roi in rois:
                    st.header("\n")
                    st.image(
                        cv2.resize(roi, (200, 100)),
                        channels="BGR",
                        use_column_width=False,
                    )

            with columns[2]:
                for confidence in confidences:
                    st.header("\n\n")
                    st.header(round(confidence, 4))
                    st.header("\n\n")
            for label in class_labels:
                if label in st.session_state.df.values:
                    st.header("Status: " + ":green[Authorized]")
                else:
                    st.header("Status: " + ":red[Unauthorized]")



else:
    st.write("Please upload an image file.")
    
