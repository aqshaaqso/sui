from ultralytics import YOLO
import cv2
import numpy as np

pose_model = YOLO("yolov8n-pose.pt")
apd_model = YOLO("best.pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    exit()

HEAD_IDXS = [0, 1, 2, 3, 4]
BODY_IDXS = [5, 6, 11, 12]
CONF_THRESHOLD = 0.5
DIST_THRESHOLD = 70

skeleton_lines = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (5, 6), (11, 12), (5, 11), (6, 12)
]

def get_center_keypoints(kp, indices):
    pts = []
    for idx in indices:
        if kp.conf[0][idx] > CONF_THRESHOLD:
            pts.append(kp.xy[0][idx].cpu().numpy())
    if pts:
        return np.mean(pts, axis=0)
    return None

def get_body_bbox(kp, indices=BODY_IDXS):
    pts = []
    for idx in indices:
        if kp.conf[0][idx] > CONF_THRESHOLD:
            pts.append(kp.xy[0][idx].cpu().numpy())
    if len(pts) < 2:
        return None
    pts = np.array(pts)
    x1, y1 = np.min(pts, axis=0)
    x2, y2 = np.max(pts, axis=0)
    return [x1, y1, x2, y2]

def get_box_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

def is_near(p1, p2, max_dist=DIST_THRESHOLD):
    return np.linalg.norm(p1 - p2) < max_dist

def is_overlap(box1, box2, threshold=0.2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = interArea / float(box1Area + box2Area - interArea + 1e-6)
    return iou > threshold

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results_pose = pose_model(frame)[0]
    results_apd = apd_model(frame)[0]

    skeletons = []
    for kp in results_pose.keypoints:
        head_center = get_center_keypoints(kp, HEAD_IDXS)
        body_bbox = get_body_bbox(kp)
        if head_center is not None:
            skeletons.append((head_center, body_bbox))
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
        box_center = get_box_center(box)
        valid = False
        for head_center, body_bbox in skeletons:
            if "helm" in label.lower() and head_center is not None:
                if is_near(box_center, head_center):
                    valid = True
                    break
            elif "rompi" in label.lower() and body_bbox is not None:
                if is_overlap(box, body_bbox):
                    valid = True
                    break
        color = (0, 255, 0) if valid else (0, 0, 255)
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {'VALID' if valid else 'INVALID'}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Validasi APD", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
