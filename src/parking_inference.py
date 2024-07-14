import cv2
import time
import argparse
from datetime import datetime
from ultralytics import YOLO
import pandas as pd


def video_inference(video_path, output_path, model, logs, parking_lot_id="A"):
    """
    Inferencing from a video file, for parking spaces detection frame by frame.
    """
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # Create VideoWriter object to save the output video

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    # Process results generator
    for result in model(video_path, stream=True, dynamic=True):
        # Get bounding boxes from YOLOv8 results
        bboxes = result.boxes
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

        # Write the frame to the output video
        out.write(frame)

        # Optional: Display the frame (uncomment to display)
        # if args["show"]:
        #     cv2.imshow('Frame', frame)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
    print(logs.head())
    logs.to_csv("parking_logs.csv", index=False)
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(
        "Output video saved at: ",
        args["output"] if args["output"] else "output_video.mp4",
    )
    print("Logs saved at: parking_logs.csv")


def image_inference(
    image_path, output_image_path, model: YOLO, logs, parking_lot_id="A"
):
    image = cv2.imread(image_path)
    # Process the image
    empty_count = 0
    occupied_count = 0
    for result in model(image, dynamic=True):
        # Example: Pass the image to another model (dummy processing here)
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

            if cls == 0:
                empty_count += 1
                class_label = "Empty"
            else:
                occupied_count += 1
                class_label = "Occupied"

            # text_width, text_height = cv2.getTextSize(class_label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            # cv2.rectangle(image, (x1, y1 - text_height*4), (x2, y1), (0, 0, 0), -1)
            # cv2.putText(image, f'{class_label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
            if cls == 0:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            print("Detected class:", class_label)
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

        print("Empty: ", empty_count)
        print("Occupied: ", occupied_count)

    if args["show"]:
        cv2.imshow("annotated", cv2.resize(image, (800, 600)))
        cv2.waitKey(0)
    # Save the logs
    cv2.imwrite(output_image_path, image)
    print("Output image saved at:", output_image_path)
    logs.to_csv("parking_logs.csv", index=False)


# Example usage
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
    model = YOLO(r"../models/parking_spaces_detection_models/best_parkingnew_openvino_model", task="detect")
    try:
        logs = pd.read_csv("parking_logs.csv")
    except:
        print("No logs found. Creating new logs file.")
        logs = pd.DataFrame(
            columns=["Time", "parking_lot", "Empty_spaces", "Occupied_spaces"]
        )
    if args["image"]:
        image_inference(args["image"], "./output.jpg", model, logs)
    if args["video"]:
        video_inference(args["video"], "./output.mp4", model, logs)

    # logs = pd.DataFrame(columns=['Time', 'License Plate', 'Confidence', 'x1', 'y1', 'x2', 'y2'])
