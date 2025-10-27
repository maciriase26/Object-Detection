 ### Technologies used

- Python 3.12
- MediaPipe Tasks (vision.object_detector)
- OpenCV
- Flask (for REST API)
- NumPy & CSV for data handling
- EfficientDet Lite2 model (.tflite)

<<<<<<< HEAD
### Example Objects Detected
=======
### Example Objects Detected (in class)
>>>>>>> b22f257b493e7e5fd759ebbf6c771fb528aca145
- Person 
- Cell phone 
- Laptop
- TV
- Chair 
- Cup

To run in terminal:
```
python3 object_detector_webcam.py --model efficientdet_lite2.tflite
```
### REST API

Run in terminal:
```
python3 api_server.py
```
 In another terminal:

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
