import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import cv2
import numpy as np
import time

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval() 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def detect_objects(image, model, threshold=0.5):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    img_tensor = F.to_tensor(img_rgb).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions = model(img_tensor)

    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    
    valid_indices = [i for i, score in enumerate(scores) if score > threshold]
    boxes = boxes[valid_indices]
    labels = labels[valid_indices]
    scores = scores[valid_indices]
    
    return boxes, labels, scores


def draw_boxes(image, boxes, scores):
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

detection_start_time = None
detection_duration = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    boxes, labels, scores = detect_objects(frame, model, threshold=0.7)
    person_indices = [i for i, label in enumerate(labels) if label == 1]
    person_boxes = boxes[person_indices]
    person_scores = scores[person_indices]

    if len(person_boxes) > 0:
        if detection_start_time is None:
            detection_start_time = time.time()
        else:
            detection_duration = time.time() - detection_start_time
    else:
        detection_start_time = None
        detection_duration = 0

    frame = draw_boxes(frame, person_boxes, person_scores)

    cv2.putText(frame, f"Waktu: {detection_duration:.2f} s",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Person Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty('Person Detection', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
