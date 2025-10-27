 ### Technologies used

- Python 3.12
- MediaPipe Tasks (vision.object_detector)
- OpenCV
- Flask (for REST API)
- NumPy & CSV for data handling
- EfficientDet Lite2 model (.tflite)

### Example Objects Detected
- Person �
- Cell phone 
- Laptop 
- Chair 
- Cup

To run in terminal-

```
python3 object_detector_webcam.py --model efficientdet_lite2.tflite
```
### REST API
```
curl -X POST http://127.0.0.1:5050/start
curl http://127.0.0.1:5050/status
curl http://127.0.0.1:5050/detections
curl -X POST http://127.0.0.1:5050/stop
```

### Logging Feature
- Logs webcam start/stop events to `system.log`
- Logs each detection (timestamp, object name, confidence score) to `detections_log.csv`
- Files are created automatically when the program runs
