import torch
print(torch.__version__)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
import cv2

cap = cv2.VideoCapture(0)  # 0 for webcam
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform inference
    results = model(frame)
    
    # Draw bounding boxes
    for bbox in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = bbox
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    
    cv2.imshow('YOLO Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
