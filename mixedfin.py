from ultralytics import YOLO
import cv2
import numpy as np

pose_model = YOLO("skeleton.pt")
apd_model = YOLO("apadah.pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    exit()

CONF_THRESHOLD = 0.5
skeleton_lines = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (5, 6), (11, 12), (5, 11), (6, 12)
]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results_pose = pose_model(frame)[0]
    results_apd = apd_model(frame)[0]

    for kp in results_pose.keypoints:
        pts = kp.xy[0].cpu().numpy()
        confs = kp.conf[0].cpu().numpy()
        for (x, y), conf in zip(pts, confs):
            if conf > CONF_THRESHOLD:
                cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 0), -1)
        for i, j in skeleton_lines:
            if confs[i] > CONF_THRESHOLD and confs[j] > CONF_THRESHOLD:
                x1, y1 = int(pts[i][0]), int(pts[i][1])
                x2, y2 = int(pts[j][0]), int(pts[j][1])
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

    for box, cls in zip(results_apd.boxes.xyxy.cpu().numpy(), results_apd.boxes.cls.cpu().numpy()):
        label = results_apd.names[int(cls)]
        x1, y1, x2, y2 = box.astype(int)
        color = (0, 0, 255) if label.startswith("no-") else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Deteksi APD", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
