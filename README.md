To run in terminal-

```
python3 object_detector_webcam.py --model efficientdet_lite2.tflite
```
### Logging Feature
- Logs webcam start/stop events to `system.log`
- Logs each detection (timestamp, object name, confidence score) to `detections_log.csv`
- Files are created automatically when the program runs
