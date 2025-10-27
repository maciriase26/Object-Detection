from flask import Flask, jsonify
import csv
import os
import threading
import subprocess

app = Flask(__name__)

CSV_FILE = "detections_log.csv"
process = None  # Will hold the running object detection process


@app.route("/detections", methods=["GET"])
def get_detections():
    """Return last 10 detections from CSV file."""
    if not os.path.exists(CSV_FILE):
        return jsonify({"message": "No detections found"}), 404

    with open(CSV_FILE, "r") as f:
        reader = list(csv.DictReader(f))
        return jsonify(reader[-10:])


@app.route("/status", methods=["GET"])
def get_status():
    """Check if the webcam detection is running."""
    running = process is not None and process.poll() is None
    return jsonify({"running": running})


@app.route("/start", methods=["POST"])
def start_detection():
    """Start the object detection process."""
    global process
    if process is None or process.poll() is not None:
        process = subprocess.Popen(["python3", "object_detector_webcam.py", "--fps"])
        return jsonify({"message": "Detection started"})
    else:
        return jsonify({"message": "Already running"})


@app.route("/stop", methods=["POST"])
def stop_detection():
    """Stop the running detection."""
    global process
    if process and process.poll() is None:
        process.terminate()
        process = None
        return jsonify({"message": "Detection stopped"})
    else:
        return jsonify({"message": "Not running"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
