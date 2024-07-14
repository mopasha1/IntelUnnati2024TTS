import cv2
from datetime import datetime
from ultralytics import YOLO
import pandas as pd
import streamlit as st
import tempfile


def read_video(file):
    # Create a temporary file to store the uploaded video
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(file.read())
    temp_file.close()

    # Open the video file with OpenCV
    cap = cv2.VideoCapture(temp_file.name)

    return cap, temp_file.name


def video_inference(
    video,
    video_path,
    model,
    frame_log_container,
    output_container,
    logs,
    parking_lot_id="A",
):
    cap = video
    count = 0
    for result in model(video_path, stream=True, dynamic=True):
        # Get bounding boxes from YOLOv8 results
        bboxes = result.boxes
        inf_speed = result.speed.get("inference")
        # print(bboxes.data)
        # Read frame from the video
        ret, frame = cap.read()
        if not ret:
            break
        empty_count = 0
        occupied_count = 0
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
            if cls == 0:
                empty_count += 1
            else:
                occupied_count += 1

            # text_width, text_height = cv2.getTextSize(class_label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            # cv2.rectangle(frame, (x1, y1 - text_height*4), (x2, y1), (0, 0, 0), -1)
            # cv2.putText(frame, f'{class_label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
            if cls == 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # print("Detected class:", class_label)
        current_time = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )  # Format: YYYY-MM-DD HH:MM:SS.mmm
        logs = pd.concat(
            [
                logs,
                pd.DataFrame(
                    {
                        "Time": current_time,
                        "parking_lot": parking_lot_id,
                        "Empty_spaces": empty_count,
                        "Occupied_spaces": occupied_count,
                    },
                    index=[0],
                ),
            ],
            ignore_index=True,
        )
        output_container.image(frame, channels="BGR")
        # Write the frame to the output video
        count += 1
        frame_log_container.subheader(
            f"Frame {count+1}: Empty spaces: {empty_count} : Occupied Spaces {occupied_count} Inference Speed: {inf_speed} ms\n"
        )

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    return logs


# Example usage
model = YOLO(r"../models/parking_spaces_detection_models/best_parkingnew_openvino_model", task="detect")

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
        logs = video_inference(
            uploaded_video, video_path, model, log_container, empty, logs=pd.DataFrame()
        )
        st.dataframe(logs)

        st.session_state["video_loaded"] = False
    except Exception as e:
        print(e)
        st.error(
            "An error occurred during inference: No parking spaces detected. Please try again with a different video file."
        )
        st.session_state["video_loaded"] = False
else:
    st.write("Please upload an video file.")
