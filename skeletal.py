import cv2
from ultralytics import YOLO
import torch

# === Load Model Pose Detection ===
model_pose = YOLO("yolov8n-pose.pt")  # Atau "bone.pt" kalau punya sendiri

# === Kamera ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Gagal membuka kamera")
    exit()

cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# === Sambungan Skeleton COCO Format ===
skeleton = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (5, 6),
    (11, 12),
    (5, 11), (6, 12)
]

# === Loop Utama ===
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Gagal membaca dari kamera")
        break

    results_pose = model_pose(frame)[0]

    for kp in results_pose.keypoints:
        pts = kp.xy[0].cpu().numpy()
        confs = kp.conf[0].cpu().numpy()

        # Gambar titik keypoints
        for (x, y), conf in zip(pts, confs):
            if conf > 0.5:
                cv2.circle(frame, (int(x), int(y)), 4, (255, 0, 0), -1)

        # Gambar garis skeleton
        for i, j in skeleton:
            if confs[i] > 0.5 and confs[j] > 0.5:
                x1, y1 = int(pts[i][0]), int(pts[i][1])
                x2, y2 = int(pts[j][0]), int(pts[j][1])
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # Tampilkan hasil
    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
