import cv2
from ultralytics import YOLO
import torch
import numpy as np
import pyrealsense2 as rs # Import library RealSense

# --- Konfigurasi ---
CAMERA_ID = 0 # Ini bisa tetap 0, tapi RealSense akan terdeteksi secara otomatis
CONFIDENCE_THRESHOLD = 0.5 
POSE_MODEL_PATH = "skeleton.pt" # Pastikan file model ini ada
WINDOW_NAME = "Skeletal Pose & Accurate Depth Detection"

# Konfigurasi Kamera RealSense (Disarankan)
# Resolusi dan FPS untuk stream RGB dan Depth
RS_WIDTH = 640
RS_HEIGHT = 480
RS_FPS = 30 # Atur sesuai kemampuan Jetson dan kamera RealSense Anda

# Tidak perlu lagi CALIBRATION_FACTOR_FOR_METERS karena RealSense memberi jarak absolut

# Sambungan Skeleton COCO Format
COCO_SKELETON_LINES = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (5, 6), (11, 12), (5, 11), (6, 12)
]

# --- Fungsi Inti ---

def load_pose_model(pose_path):
    """Memuat model deteksi pose."""
    model_pose = YOLO(pose_path) 
    return model_pose

def initialize_realsense_camera(width, height, fps):
    """Menginisialisasi kamera Intel RealSense."""
    pipeline = rs.pipeline()
    config = rs.config()

    # Aktifkan stream RGB
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    # Aktifkan stream Depth
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    # Mulai pipeline
    profile = pipeline.start(config)
    print(f"✅ RealSense Camera initialized with RGB: {width}x{height}@{fps}fps, Depth: {width}x{height}@{fps}fps")
    return pipeline, profile

def draw_pose_annotations(frame, results_pose, conf_thresh, skeleton_lines, circle_radius, line_thickness):
    """Menggambar keypoints dan skeleton pada frame."""
    if results_pose.keypoints and len(results_pose.keypoints.xy) > 0:
        for kp in results_pose.keypoints:
            if kp.conf is not None and len(kp.conf) > 0:
                pts = kp.xy[0].cpu().numpy()
                confs = kp.conf[0].cpu().numpy()
            else:
                continue

            for i, (x, y) in enumerate(pts):
                if confs[i] > conf_thresh:
                    cv2.circle(frame, (int(x), int(y)), circle_radius, (255, 0, 0), -1)

            for i, j in skeleton_lines:
                if confs[i] > conf_thresh and confs[j] > conf_thresh:
                    x1, y1 = int(pts[i][0]), int(pts[i][1])
                    x2, y2 = int(pts[j][0]), int(pts[j][1])
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), line_thickness)
    return frame

def get_accurate_depth_and_annotate(color_frame, depth_frame, results_pose):
    """Mengambil data kedalaman akurat dari RealSense dan menambahkan anotasi jarak."""
    # Konversi frame RealSense ke array NumPy
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # Konversi depth_image (mm) ke meter atau cm untuk visualisasi
    # RealSense umumnya memberikan jarak dalam milimeter (mm)
    depth_scale = depth_frame.get_units() # Dapatkan unit skala kedalaman (biasanya 0.001 meter/1mm per unit)

    annotated_frame = color_image.copy() # Gunakan salinan untuk anotasi

    if results_pose.boxes and len(results_pose.boxes.xyxy) > 0:
        for box in results_pose.boxes.xyxy: 
            x1, y1, x2, y2 = map(int, box[:4])
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(color_image.shape[1], x2)
            y2 = min(color_image.shape[0], y2)

            if x2 > x1 and y2 > y1:
                # Ambil bagian depth map di dalam bounding box
                depth_roi = depth_image[y1:y2, x1:x2]
                
                if depth_roi.size > 0:
                    # Filter out zero values (invalid depth, e.g., beyond range or no detection)
                    valid_depths = depth_roi[depth_roi > 0] 
                    
                    if valid_depths.size > 0:
                        # Hitung median dari kedalaman yang valid
                        distance_mm = np.median(valid_depths)
                        distance_meters = distance_mm * depth_scale # Konversi ke meter
                        
                        label_distance = f"Jarak: {distance_meters:.2f} m"
                    else:
                        label_distance = f"Jarak: N/A" # No valid depth detected
                    
                    # Gambar bounding box dan label jarak
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2) 
                    cv2.putText(annotated_frame, label_distance, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return annotated_frame

# --- Main Program ---
def main():
    # Load model pose
    model_pose = load_pose_model(POSE_MODEL_PATH)
    
    # Inisialisasi kamera RealSense
    pipeline, profile = initialize_realsense_camera(RS_WIDTH, RS_HEIGHT, RS_FPS)
    
    # Dapatkan stream profile untuk alignment (penting jika resolusi RGB dan Depth berbeda)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale() # Dapatkan skala kedalaman

    # Buat objek alignment untuk menyamakan resolusi Depth ke RGB
    align_to_color = rs.align(rs.stream.color)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        while True:
            # Tunggu frame RealSense (RGB dan Depth)
            frames = pipeline.wait_for_frames()
            
            # Lakukan alignment antara depth frame dan color frame
            aligned_frames = align_to_color.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue # Skip frame jika salah satu tidak ada

            # Konversi color_frame ke NumPy array untuk YOLO
            color_image_np = np.asanyarray(color_frame.get_data())

            # Jalankan inferensi pose pada gambar RGB
            results_pose = model_pose(color_image_np, verbose=False)[0] 

            # Dapatkan kedalaman akurat dan anotasi
            final_annotated_frame = get_accurate_depth_and_annotate(
                color_frame, # Pass RealSense color_frame object
                depth_frame, # Pass RealSense depth_frame object
                results_pose
            )
            
            # Gambar anotasi pose (keypoint dan skeleton) pada frame yang sudah ada anotasi jarak
            # Pastikan KEYPOINT_CIRCLE_RADIUS dan SKELETON_LINE_THICKNESS terdefinisi
            KEYPOINT_CIRCLE_RADIUS = 4 # Definisi lokal atau global
            SKELETON_LINE_THICKNESS = 2 # Definisi lokal atau global
            
            final_annotated_frame = draw_pose_annotations(
                final_annotated_frame, 
                results_pose,
                CONFIDENCE_THRESHOLD,
                COCO_SKELETON_LINES,
                KEYPOINT_CIRCLE_RADIUS,
                SKELETON_LINE_THICKNESS
            )

            cv2.imshow(WINDOW_NAME, final_annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Hentikan pipeline RealSense saat selesai atau ada error
        pipeline.stop()
        cv2.destroyAllWindows()
        print("RealSense pipeline stopped and windows closed.")

if __name__ == "__main__":
    # Menambahkan definisi variabel global untuk draw_pose_annotations
    # Ini diperlukan karena saya menghapus mereka sebelumnya
    KEYPOINT_CIRCLE_RADIUS = 4
    SKELETON_LINE_THICKNESS = 2
    main()