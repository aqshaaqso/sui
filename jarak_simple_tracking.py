import cv2
import numpy as np
from collections import defaultdict
import time

CAMERA_ID = 0
CONFIDENCE_THRESHOLD = 0.5 

POSE_MODEL_PATH = "skeleton.pt" 
WINDOW_NAME = "Deteksi Jarak dengan Simple Tracking"

CALIBRATION_FACTOR_GEOMETRIC = 137.70
DISTANCE_FILTER_ENABLED = True
MAX_DISTANCE_METERS = 5.2 

BOX_AND_TEXT_COLOR = (255, 0, 255)  # Magenta / Pink Terang
TRACK_COLOR = (0, 255, 0)  # Green for tracked objects

RIGHT_SHOULDER_IDX = 5
LEFT_SHOULDER_IDX = 6

COCO_SKELETON_LINES = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (5, 6), (11, 12), (5, 11), (6, 12)
]
KEYPOINT_CIRCLE_RADIUS = 4
SKELETON_LINE_THICKNESS = 2

# Simple Tracker parameters
MAX_TRACKERS = 10
TRACKER_INACTIVE_FRAMES = 15
DISTANCE_THRESHOLD = 100  # pixels

class SimpleTracker:
    """Simple centroid-based tracker as fallback"""
    def __init__(self, bbox, tracker_id):
        self.tracker_id = tracker_id
        self.active = True
        self.inactive_frames = 0
        self.last_bbox = bbox
        self.distance_history = []
        self.creation_time = time.time()
        
        # Calculate centroid
        x1, y1, x2, y2 = bbox
        self.centroid = [(x1 + x2) / 2, (y1 + y2) / 2]
        self.last_centroid = self.centroid.copy()
    
    def update(self, detections):
        """Update tracker with new detections"""
        if not self.active:
            return False, None
        
        best_match = None
        min_distance = float('inf')
        
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            centroid = [(x1 + x2) / 2, (y1 + y2) / 2]
            
            distance = np.sqrt((centroid[0] - self.last_centroid[0])**2 + 
                             (centroid[1] - self.last_centroid[1])**2)
            
            if distance < min_distance and distance < DISTANCE_THRESHOLD:
                min_distance = distance
                best_match = detection
        
        if best_match is not None:
            bbox = best_match['bbox']
            x1, y1, x2, y2 = bbox
            self.centroid = [(x1 + x2) / 2, (y1 + y2) / 2]
            self.last_centroid = self.centroid.copy()
            self.last_bbox = bbox
            self.inactive_frames = 0
            return True, bbox
        else:
            self.inactive_frames += 1
            if self.inactive_frames > TRACKER_INACTIVE_FRAMES:
                self.active = False
            return False, self.last_bbox
    
    def add_distance(self, distance):
        """Add distance measurement to history"""
        self.distance_history.append(distance)
        if len(self.distance_history) > 10:
            self.distance_history.pop(0)
    
    def get_smoothed_distance(self):
        """Get smoothed distance using moving average"""
        if not self.distance_history:
            return None
        return np.mean(self.distance_history)

class SimpleTrackingSystem:
    """Simple tracking system using centroid-based tracking"""
    
    def __init__(self):
        self.trackers = {}
        self.next_id = 0
        self.frame_count = 0
    
    def update_trackers(self, detections):
        """Update all trackers and match with new detections"""
        self.frame_count += 1
        
        active_trackers = {}
        matched_detections = []
        
        for tracker_id, tracker in self.trackers.items():
            success, bbox = tracker.update(detections)
            if tracker.active:
                active_trackers[tracker_id] = tracker
                
                if success:
                    for detection in detections:
                        if detection['bbox'] == bbox:
                            detection['tracker_id'] = tracker_id
                            matched_detections.append(detection)
                            if 'distance' in detection:
                                tracker.add_distance(detection['distance'])
                            break
        
        self.trackers = active_trackers
        
        for detection in detections:
            if detection not in matched_detections and len(self.trackers) < MAX_TRACKERS:
                new_tracker = SimpleTracker(detection['bbox'], self.next_id)
                self.trackers[self.next_id] = new_tracker
                detection['tracker_id'] = self.next_id
                if 'distance' in detection:
                    new_tracker.add_distance(detection['distance'])
                self.next_id += 1
        
        return detections
    
    def get_tracked_objects(self):
        """Get current tracked objects with their information"""
        tracked_objects = []
        for tracker_id, tracker in self.trackers.items():
            if tracker.active:
                smoothed_distance = tracker.get_smoothed_distance()
                tracked_objects.append({
                    'tracker_id': tracker_id,
                    'bbox': tracker.last_bbox,
                    'distance': smoothed_distance,
                    'age': time.time() - tracker.creation_time
                })
        return tracked_objects

def initialize_camera(camera_id):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Kamera ID {camera_id} tidak dapat dibuka.")
        exit()
    return cap

def process_frame_with_simple_tracking(frame, model_pose, tracking_system):
    """Process frame with simple tracking integration"""
    results_pose = model_pose(frame, verbose=False)[0]
    
    detections = []
    pixel_width_for_calibration = 0.0

    if results_pose.boxes and results_pose.keypoints:
        for i in range(len(results_pose.boxes)):
            kp = results_pose.keypoints[i]
            if kp.conf is None or len(kp.conf) == 0: 
                continue

            pts = kp.xy[0].cpu().numpy()
            confs = kp.conf[0].cpu().numpy()

            if confs[LEFT_SHOULDER_IDX] > CONFIDENCE_THRESHOLD and confs[RIGHT_SHOULDER_IDX] > CONFIDENCE_THRESHOLD:
                left_shoulder_pt = pts[LEFT_SHOULDER_IDX]
                right_shoulder_pt = pts[RIGHT_SHOULDER_IDX]

                pixel_width = np.linalg.norm(left_shoulder_pt - right_shoulder_pt)
                pixel_width_for_calibration = pixel_width 

                if pixel_width > 0:
                    distance_in_meters = CALIBRATION_FACTOR_GEOMETRIC / pixel_width
                    
                    if not DISTANCE_FILTER_ENABLED or distance_in_meters <= MAX_DISTANCE_METERS:
                        box = results_pose.boxes[i]
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        
                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'keypoints': pts,
                            'confidences': confs,
                            'distance': distance_in_meters,
                            'pixel_width': pixel_width
                        }
                        detections.append(detection)

    tracked_detections = tracking_system.update_trackers(detections)
    
    for detection in tracked_detections:
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        tracker_id = detection.get('tracker_id')
        
        color = TRACK_COLOR if tracker_id is not None else BOX_AND_TEXT_COLOR
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        labels = []
        if 'distance' in detection:
            labels.append(f"Jarak: {detection['distance']:.2f} m")
        if tracker_id is not None:
            labels.append(f"ID: {tracker_id}")
        
        for idx, label in enumerate(labels):
            label_y = y1 - 10 - (idx * 25)
            cv2.putText(frame, label, (x1, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if 'keypoints' in detection and 'confidences' in detection:
            pts = detection['keypoints']
            confs = detection['confidences']
            
            # Draw keypoints
            for j, (x, y) in enumerate(pts):
                if confs[j] > CONFIDENCE_THRESHOLD:
                    cv2.circle(frame, (int(x), int(y)), KEYPOINT_CIRCLE_RADIUS, (255, 0, 0), -1)
            
            # Draw skeleton
            for p1_idx, p2_idx in COCO_SKELETON_LINES:
                if confs[p1_idx] > CONFIDENCE_THRESHOLD and confs[p2_idx] > CONFIDENCE_THRESHOLD:
                    p1_x, p1_y = int(pts[p1_idx][0]), int(pts[p1_idx][1])
                    p2_x, p2_y = int(pts[p2_idx][0]), int(pts[p2_idx][1])
                    cv2.line(frame, (p1_x, p1_y), (p2_x, p2_y), (0, 255, 255), SKELETON_LINE_THICKNESS)
    
    return frame, pixel_width_for_calibration, len(tracking_system.trackers)

def main():
    # Load YOLO model
    from ultralytics import YOLO
    import torch
    
    print("Memuat model YOLO... Harap tunggu.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan device: {device}")
    model_pose = YOLO(POSE_MODEL_PATH)
    
    if model_pose is None:
        print("Error: Model pose tidak tersedia.")
        return
    
    cap = initialize_camera(CAMERA_ID)
    tracking_system = SimpleTrackingSystem()
    
    print("Tekan 'q' untuk keluar")
    print("Tekan 'c' untuk mode kalibrasi")
    print("Tekan 'r' untuk reset semua tracker")
    print("Tekan 's' untuk menampilkan statistik tracking")

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: 
            break

        frame_count += 1
        
        # Process frame with simple tracking
        annotated_frame, pixel_width, num_trackers = process_frame_with_simple_tracking(frame, model_pose, tracking_system)
        
        # Add tracking statistics to frame
        stats_text = f"Trackers: {num_trackers} | Frame: {frame_count}"
        cv2.putText(annotated_frame, stats_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow(WINDOW_NAME, annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):  # Calibration mode
            if pixel_width > 0:
                print(f"\n[MODE KALIBRASI] Tekan 'c' saat objek tepat di jarak yang diketahui.")
                print(f"==> Lebar Bahu terukur (pixel): {pixel_width:.2f}")
                print(f"==> Hitung Faktor Kalibrasi Anda: (Lebar Bahu Pixel) x (Jarak Sebenarnya dalam Meter)")
                print(f"==> Contoh: Jika jarak Anda 2 meter, faktor baru = {pixel_width:.2f} x 2.0 = {pixel_width * 2.0:.2f}\n")
            else:
                print("\n[MODE KALIBRASI] Bahu tidak terdeteksi.\n")
        elif key == ord("r"):  # Reset all trackers
            tracking_system = SimpleTrackingSystem()
            print("\n[RESET] Semua tracker telah direset.\n")
        elif key == ord("s"):  # Show statistics
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            print(f"\n[STATISTIK]")
            print(f"FPS: {fps:.2f}")
            print(f"Total Frames: {frame_count}")
            print(f"Active Trackers: {num_trackers}")
            print(f"Total Tracker IDs Used: {tracking_system.next_id}")
            print()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()