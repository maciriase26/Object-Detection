 ### Technologies used

- Python 3.12
- MediaPipe Tasks (vision.object_detector)
- OpenCV
- Flask (for REST API)
- NumPy & CSV for data handling
- EfficientDet Lite2 model (.tflite)


### Example Objects Detected
https://github.com/maciriase26/Object-Detection/blob/b8d5ae1b7494a6fd3a4801f25885d7f6d72dee6f/images/Example.png

=======
### Example Objects Detected (in class)
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
