import cv2
import numpy as np
from ultralytics import YOLO

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    vector_ba = a - b
    vector_bc = c - b
    dot_product = np.dot(vector_ba, vector_bc)
    norm_ba = np.linalg.norm(vector_ba)
    norm_bc = np.linalg.norm(vector_bc)
    if norm_ba == 0 or norm_bc == 0:
        return 180.0
    cosine_angle = np.clip(dot_product / (norm_ba * norm_bc), -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

COCO_SKELETON_LINES = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (5, 6), (11, 12), (5, 11), (6, 12)
]
KEYPOINT_CIRCLE_RADIUS = 4
SKELETON_LINE_THICKNESS = 2

model = YOLO('skeleton.pt')

cap = cv2.VideoCapture(0)

RUNNING_THRESHOLD_ANGLE = 90.0
STILLNESS_THRESHOLD = 25.0

CALIBRATION_FACTOR_GEOMETRIC = 137.70
CONFIDENCE_THRESHOLD = 0.5
RIGHT_SHOULDER_IDX = 5
LEFT_SHOULDER_IDX = 6

MOVEMENT_HISTORY_SIZE = 5
SMOOTHING_FACTOR = 0.3

TRACKED_KEYPOINTS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

prev_person_keypoints = None
movement_history = []
smoothed_movement = 0.0
individual_keypoint_movements = {}

FONT = cv2.FONT_HERSHEY_SIMPLEX
RUN_STATUS_COLOR = (0, 255, 0)
STILL_STATUS_COLOR = (255, 255, 0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video selesai atau terjadi error. Program berhenti.")
        break

    results = model(frame, verbose=False)
    annotated_frame = frame.copy()

    run_status = "N/A"
    still_status = "N/A"

    if results[0].keypoints and len(results[0].keypoints.data) > 0:
        current_person_keypoints = results[0].keypoints.data[0]

        left_hip = current_person_keypoints[11][:2].cpu().numpy()
        left_knee = current_person_keypoints[13][:2].cpu().numpy()
        left_ankle = current_person_keypoints[15][:2].cpu().numpy()
        right_hip = current_person_keypoints[12][:2].cpu().numpy()
        right_knee = current_person_keypoints[14][:2].cpu().numpy()
        right_ankle = current_person_keypoints[16][:2].cpu().numpy()

        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

        distance_in_meters = 0.0
        left_shoulder = current_person_keypoints[LEFT_SHOULDER_IDX][:2].cpu().numpy()
        right_shoulder = current_person_keypoints[RIGHT_SHOULDER_IDX][:2].cpu().numpy()
        
        if (current_person_keypoints[LEFT_SHOULDER_IDX][2] > CONFIDENCE_THRESHOLD and 
            current_person_keypoints[RIGHT_SHOULDER_IDX][2] > CONFIDENCE_THRESHOLD):
            pixel_width = np.linalg.norm(left_shoulder - right_shoulder)
            if pixel_width > 0:
                distance_in_meters = CALIBRATION_FACTOR_GEOMETRIC / pixel_width

        total_movement = 0.0
        detected_keypoints_count = 0
        individual_keypoint_movements = {}
        
        if prev_person_keypoints is not None:
            for i in TRACKED_KEYPOINTS:
                if (current_person_keypoints[i][2] > CONFIDENCE_THRESHOLD and 
                    prev_person_keypoints[i][2] > CONFIDENCE_THRESHOLD):
                    current_point = current_person_keypoints[i][:2].cpu().numpy()
                    prev_point = prev_person_keypoints[i][:2].cpu().numpy()
                    keypoint_movement = np.linalg.norm(current_point - prev_point)
                    individual_keypoint_movements[i] = keypoint_movement
                    total_movement += keypoint_movement
                    detected_keypoints_count += 1
            
            if detected_keypoints_count > 0:
                movement_history.append(total_movement)
                if len(movement_history) > MOVEMENT_HISTORY_SIZE:
                    movement_history.pop(0)
                
                smoothed_movement = SMOOTHING_FACTOR * total_movement + (1 - SMOOTHING_FACTOR) * smoothed_movement
                
                avg_movement = sum(movement_history) / len(movement_history)
                is_moving = avg_movement >= STILLNESS_THRESHOLD
            else:
                is_moving = False
        else:
            is_moving = False

        if is_moving and (left_knee_angle < RUNNING_THRESHOLD_ANGLE or right_knee_angle < RUNNING_THRESHOLD_ANGLE):
            run_status = "Berlari!"
            RUN_STATUS_COLOR = (0, 0, 255)
        elif is_moving:
            run_status = "Berjalan"
            RUN_STATUS_COLOR = (0, 165, 255)
        else:
            run_status = "Tidak Bergerak"
            RUN_STATUS_COLOR = (0, 255, 0)

        if prev_person_keypoints is not None:
            if smoothed_movement < STILLNESS_THRESHOLD:
                still_status = "Diam / Fokus"
                STILL_STATUS_COLOR = (255, 255, 0)
            else:
                still_status = "Bergerak"
                STILL_STATUS_COLOR = (0, 255, 255)
        else:
            still_status = "Mengkalibrasi..."

        prev_person_keypoints = current_person_keypoints

        annotated_frame = frame.copy()
        
        for i, keypoint in enumerate(current_person_keypoints):
            if keypoint[2] > 0.5:
                x, y = int(keypoint[0]), int(keypoint[1])
                cv2.circle(annotated_frame, (x, y), KEYPOINT_CIRCLE_RADIUS, (255, 0, 0), -1)
        
        for p1_idx, p2_idx in COCO_SKELETON_LINES:
            if p1_idx not in {0, 1, 2, 3, 4} and p2_idx not in {0, 1, 2, 3, 4}:
                if current_person_keypoints[p1_idx][2] > 0.5 and current_person_keypoints[p2_idx][2] > 0.5:
                    p1 = current_person_keypoints[p1_idx][:2].cpu().numpy()
                    p2 = current_person_keypoints[p2_idx][:2].cpu().numpy()
                    cv2.line(annotated_frame, tuple(p1.astype(int)), tuple(p2.astype(int)), (0, 255, 255), SKELETON_LINE_THICKNESS)

        cv2.putText(annotated_frame, f"Smoothed Movement: {smoothed_movement:.2f}", (20, 150), FONT, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Detected Keypoints: {detected_keypoints_count}", (20, 180), FONT, 0.8, (255, 255, 255), 2)
        if distance_in_meters > 0:
            cv2.putText(annotated_frame, f"Jarak: {distance_in_meters:.2f} m", (20, 210), FONT, 0.8, (255, 0, 255), 2)
        
        if individual_keypoint_movements:
            sorted_movements = sorted(individual_keypoint_movements.items(), key=lambda x: x[1], reverse=True)[:3]
            for idx, (keypoint_idx, movement) in enumerate(sorted_movements):
                cv2.putText(annotated_frame, f"KP{keypoint_idx}: {movement:.1f}", (20, 240 + idx * 25), FONT, 0.6, (0, 255, 255), 1)

    else:
        prev_person_keypoints = None
        run_status = "Tidak ada orang"
        still_status = "Tidak ada orang"
        annotated_frame = frame.copy()

    cv2.putText(annotated_frame, f"Status Lari: {run_status}", (20, 50), FONT, 1, RUN_STATUS_COLOR, 2, cv2.LINE_AA)
    cv2.putText(annotated_frame, f"Status Gerak: {still_status}", (20, 100), FONT, 1, STILL_STATUS_COLOR, 2, cv2.LINE_AA)

    cv2.imshow("Multi-Detector AI", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()