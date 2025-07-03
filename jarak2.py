import cv2
from ultralytics import YOLO
import torch
import numpy as np

# --- Konfigurasi ---
CAMERA_ID = 0
CONFIDENCE_THRESHOLD = 0.5 

POSE_MODEL_PATH = "skeleton.pt" 
WINDOW_NAME = "Deteksi Jarak Geometri (Tanpa MiDaS)"

CALIBRATION_FACTOR_GEOMETRIC = 137.70
# --- Konfigurasi Filter Jarak ---
DISTANCE_FILTER_ENABLED = True
MAX_DISTANCE_METERS = 5.2 

# --- Konfigurasi Visualisasi ---
BOX_AND_TEXT_COLOR = (255, 0, 255) # Magenta / Pink Terang

RIGHT_SHOULDER_IDX = 5
LEFT_SHOULDER_IDX = 6

COCO_SKELETON_LINES = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (5, 6), (11, 12), (5, 11), (6, 12)
]
KEYPOINT_CIRCLE_RADIUS = 4
SKELETON_LINE_THICKNESS = 2

def load_yolo_model(pose_path):
    print("Memuat model YOLO... Harap tunggu.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan device: {device}")
    model_pose = YOLO(pose_path) 
    return model_pose, device

def initialize_camera(camera_id):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Kamera ID {camera_id} tidak dapat dibuka.")
        exit()
    return cap

def process_frame_geometric(frame, model_pose):
    results_pose = model_pose(frame, verbose=False)[0]
    
    pixel_width_for_calibration = 0.0

    # Pastikan ada orang yang terdeteksi
    if results_pose.boxes and results_pose.keypoints:
        # Loop untuk setiap orang yang terdeteksi, menggunakan index 'i'
        for i in range(len(results_pose.boxes)):
            
            # Ambil data rangka (keypoints) untuk orang ke-'i'
            kp = results_pose.keypoints[i]
            if kp.conf is None or len(kp.conf) == 0: continue

            pts = kp.xy[0].cpu().numpy()
            confs = kp.conf[0].cpu().numpy()

            # Pastikan kedua bahu terdeteksi dengan confidence yang cukup
            if confs[LEFT_SHOULDER_IDX] > CONFIDENCE_THRESHOLD and confs[RIGHT_SHOULDER_IDX] > CONFIDENCE_THRESHOLD:
                # Ambil koordinat kedua bahu
                left_shoulder_pt = pts[LEFT_SHOULDER_IDX]
                right_shoulder_pt = pts[RIGHT_SHOULDER_IDX]

                # Hitung jarak antar bahu dalam piksel
                pixel_width = np.linalg.norm(left_shoulder_pt - right_shoulder_pt)
                pixel_width_for_calibration = pixel_width # Simpan untuk mode kalibrasi

                if pixel_width > 0:
                    # Hitung jarak menggunakan faktor kalibrasi
                    distance_in_meters = CALIBRATION_FACTOR_GEOMETRIC / pixel_width
                    
                    if not DISTANCE_FILTER_ENABLED or distance_in_meters <= MAX_DISTANCE_METERS:
                        label_distance = f"Jarak: {distance_in_meters:.2f} m"

                        # Ambil data kotak (bounding box) untuk orang ke-'i' dari 'results_pose.boxes'
                        box = results_pose.boxes[i]
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

                        # Gambar semua anotasi
                        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_AND_TEXT_COLOR, 2)
                        cv2.putText(frame, label_distance, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BOX_AND_TEXT_COLOR, 2)
                        
                        # Gambar skeleton
                        for j, (x, y) in enumerate(pts):
                            if confs[j] > CONFIDENCE_THRESHOLD:
                                cv2.circle(frame, (int(x), int(y)), KEYPOINT_CIRCLE_RADIUS, (255, 0, 0), -1)
                        for p1_idx, p2_idx in COCO_SKELETON_LINES:
                            if confs[p1_idx] > CONFIDENCE_THRESHOLD and confs[p2_idx] > CONFIDENCE_THRESHOLD:
                                p1_x, p1_y = int(pts[p1_idx][0]), int(pts[p1_idx][1])
                                p2_x, p2_y = int(pts[p2_idx][0]), int(pts[p2_idx][1])
                                cv2.line(frame, (p1_x, p1_y), (p2_x, p2_y), (0, 255, 255), SKELETON_LINE_THICKNESS)
    
    return frame, pixel_width_for_calibration

def main():
    model_pose, _ = load_yolo_model(POSE_MODEL_PATH)
    cap = initialize_camera(CAMERA_ID)

    while True:
        ret, frame = cap.read()
        if not ret: break

        annotated_frame, pixel_width = process_frame_geometric(frame, model_pose)
        
        cv2.imshow(WINDOW_NAME, annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"): # FITUR MODE KALIBRASI
            if pixel_width > 0:
                print(f"\n[MODE KALIBRASI] Tekan 'c' saat objek tepat di jarak yang diketahui.")
                print(f"==> Lebar Bahu terukur (pixel): {pixel_width:.2f}")
                print(f"==> Hitung Faktor Kalibrasi Anda: (Lebar Bahu Pixel) x (Jarak Sebenarnya dalam Meter)")
                print(f"==> Contoh: Jika jarak Anda 2 meter, faktor baru = {pixel_width:.2f} x 2.0 = {pixel_width * 2.0:.2f}\n")
            else:
                print("\n[MODE KALIBRASI] Bahu tidak terdeteksi.\n")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()