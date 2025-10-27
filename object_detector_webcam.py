import time
import urllib
import pathlib
import argparse

import numpy as np
import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Logging setup 
import logging, csv, os
from datetime import datetime

stop_program = False

logging.basicConfig(
    filename="system.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

CSV_FILE = "detections_log.csv"

def _init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp_iso", "object", "score"])

def _log_webcam(event: str):
    logging.info(f"Webcam {event}")

def _log_detection(ts_ms: int, obj_name: str, score: float):
    ts_iso = datetime.fromtimestamp(ts_ms / 1000.0).isoformat()
    with open(CSV_FILE, "a", newline="") as f:
        csv.writer(f).writerow([ts_iso, obj_name, round(float(score), 4)])

# ----------------------------
# Drawing constants
# ----------------------------
MARGIN = 30  # pixels
ROW_SIZE = 30  # pixels
FONT_SIZE = 3
FONT_THICKNESS = 2
RECT_COLOR = (255, 0, 0)  # blue (BGR)
TEXT_COLOR = (255, 255, 255)  # white
FPS_AVG_FRAME_COUNT = 10


def visualize(image, detection_result) -> np.ndarray:
    """Draws bounding boxes on the input image and return it."""
    for detection in detection_result.detections:

        # Draw bounding box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, RECT_COLOR, 3)

        # Draw label and score
        category = detection.categories[0]
        category_name = getattr(category, "category_name", "unknown")
        probability = round(float(getattr(category, "score", 0.0)), 2)
        result_text = f"{category_name} ({probability})"
        text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(
            image,
            result_text,
            text_location,
            cv2.FONT_HERSHEY_PLAIN,
            FONT_SIZE,
            TEXT_COLOR,
            FONT_THICKNESS,
        )

    return image


def download(url, model_path):
    print()
    print(f"Downloading model from {url}...")
    urllib.request.urlretrieve(url, model_path)
    print(f"Model downloaded and saved as {model_path}")


def show_fps(
    current_frame,
    fps,
    row_size=40,        # pixels
    left_margin=24,     # pixels
    text_color=(20, 60, 220),  # crimson (BGR)
    font_size=2,
    font_thickness=2,
):
    # Show the FPS
    fps_text = "FPS = {:.1f}".format(fps)
    text_location = (left_margin, row_size)
    cv2.putText(
        current_frame,
        fps_text,
        text_location,
        cv2.FONT_HERSHEY_PLAIN,
        font_size,
        text_color,
        font_thickness,
    )

def click_button(event, x, y, flags, param):
    global stop_program
    
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon = np.array([[(20,20), (150, 20), (150, 60), (20, 60)]])
        
        is_inside = cv2.pointPolygonTest(polygon, (x, y), False)
        if is_inside > 0:
            stop_program = True

def run(args):
    """Continuously run inference on images acquired from the camera."""
    # Ensure model exists or download it
    if not args.model.exists():
        download(args.url, args.model)

    # Buffer for async results
    detection_result_list = []

    def visualize_callback(
        result: vision.ObjectDetectorResult, output_image: mp.Image, timestamp_ms: int
    ):
        # Keep the timestamp with the result so we can log it later
        result.timestamp_ms = timestamp_ms
        detection_result_list.append(result)

    # Initialize ObjectDetector
    base_options = python.BaseOptions(model_asset_path=args.model)
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        score_threshold=0.5,
        result_callback=visualize_callback,
    )
    detector = vision.ObjectDetector.create_from_options(options)

    # FPS calculation state
    counter, fps = 0, 0
    start_time = time.time()

    # Open webcam
    cap = cv2.VideoCapture(args.camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.frame_height)

    # Initialize logs
    _init_csv()
    _log_webcam("started")

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to grab frame")
                break

            # FPS counter
            counter += 1

            # Mirror effect
            frame = cv2.flip(frame, 1)

            # Convert to RGB as expected by MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Run object detection asynchronously
            detector.detect_async(mp_image, counter)
            
          

            # Convert back to BGR for OpenCV display
            current_frame = mp_image.numpy_view()
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)
            
            polygon = np.array([[(20,20), (150, 20), (150, 60), (20, 60)]])
            cv2.fillPoly(current_frame, polygon, (0, 0, 200))  # Dark blue button
            cv2.putText(current_frame, "EXIT", (30,60), cv2.FONT_HERSHEY_PLAIN, 3, (245,245,245), 3)
            
            cv2.setMouseCallback("Object Detection", click_button)


            # Optional FPS overlay
            if args.fps:
                if counter % FPS_AVG_FRAME_COUNT == 0:
                    end_time = time.time()
                    fps = FPS_AVG_FRAME_COUNT / (end_time - start_time)
                    start_time = time.time()
                show_fps(current_frame, fps)

            if detection_result_list:
                # Grab the latest result
                result = detection_result_list[0]

                # ---- Log detections (added) ----
                ts_ms = getattr(result, "timestamp_ms", int(time.time() * 1000))
                for d in result.detections:
                    category = d.categories[0]
                    obj_name = getattr(category, "category_name", "unknown")
                    score = float(getattr(category, "score", 0.0))
                    _log_detection(ts_ms, obj_name, score)

                # Visualize and show
                annotated_frame = visualize(current_frame, result)
                cv2.imshow("Object Detection", annotated_frame)
                detection_result_list.clear()
            else:
                cv2.imshow("Object Detection", current_frame)

            # Exit on 'q'
            if cv2.waitKey(5) & 0xFF == ord("q") or stop_program:
                break

    finally:
        # Always log stop, even on exceptions/Ctrl-C
        _log_webcam("stopped")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model",
        help=(
            "Path of the model. Default: `efficientdet.tflite`. If not found, the "
            "script will attempt to download it from the default url."
        ),
        required=False,
        default="efficientdet.tflite",
    )

    parser.add_argument(
        "--url",
        help=(
            "Model url, found on the appropriate page here:\n"
            "https://ai.google.dev/edge/mediapipe/solutions"
        ),
        required=False,
        default="https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite",
    )

    parser.add_argument(
        "--camera_id", help="Id of camera.", required=False, type=int, default=0
    )

    parser.add_argument(
        "--frame_width",
        help="Width of frame to capture from camera.",
        required=False,
        type=int,
        default=1280,
    )

    parser.add_argument(
        "--frame_height",
        help="Height of frame to capture from camera.",
        required=False,
        type=int,
        default=720,
    )

    parser.add_argument(
        "--fps",
        help="Whether to display the FPS on the screen",
        required=False,
        action="store_true",
    )

    args = parser.parse_args()
    args.camera_id = int(args.camera_id)
    args.model = pathlib.Path(args.model)

    run(args)
