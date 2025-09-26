import cv2
import numpy as np
from collections import defaultdict
import time

# Demo parameters
CAMERA_ID = 0
WINDOW_NAME = "KCF Tracking Demo"
MAX_TRACKERS = 5

class SimpleKCFDemo:
    """Simple KCF tracking demonstration"""
    
    def __init__(self):
        self.trackers = {}
        self.next_id = 0
        self.selecting = False
        self.selection_start = None
        self.current_selection = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for manual object selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selecting = True
            self.selection_start = (x, y)
            self.current_selection = None
            
        elif event == cv2.EVENT_MOUSEMOVE and self.selecting:
            self.current_selection = (self.selection_start[0], self.selection_start[1], 
                                    x - self.selection_start[0], y - self.selection_start[1])
            
        elif event == cv2.EVENT_LBUTTONUP:
            if self.selecting and self.current_selection:
                x, y, w, h = self.current_selection
                if w > 10 and h > 10:  # Minimum size check
                    self.add_tracker(param, (x, y, abs(w), abs(h)))
            self.selecting = False
            self.current_selection = None
    
    def add_tracker(self, frame, bbox):
        """Add new KCF tracker"""
        if len(self.trackers) >= MAX_TRACKERS:
            print("Maximum number of trackers reached!")
            return
        
        # Try different tracker creation methods based on OpenCV version
        tracker = None
        try:
            # Try newer OpenCV API first
            tracker = cv2.legacy.TrackerKCF_create()
        except AttributeError:
            try:
                # Try older OpenCV API
                tracker = cv2.TrackerKCF_create()
            except AttributeError:
                try:
                    # Try alternative tracker (CSRT)
                    tracker = cv2.legacy.TrackerCSRT_create()
                    print("Using CSRT tracker (KCF not available)")
                except AttributeError:
                    try:
                        tracker = cv2.TrackerCSRT_create()
                        print("Using CSRT tracker (KCF not available)")
                    except AttributeError:
                        print("No suitable tracker available!")
                        return
        
        success = tracker.init(frame, bbox)
        
        if success:
            self.trackers[self.next_id] = {
                'tracker': tracker,
                'bbox': bbox,
                'active': True,
                'age': 0
            }
            print(f"Added tracker {self.next_id} at {bbox}")
            self.next_id += 1
        else:
            print("Failed to initialize tracker!")
    
    def update_trackers(self, frame):
        """Update all active trackers"""
        active_trackers = {}
        
        for tracker_id, tracker_info in self.trackers.items():
            if not tracker_info['active']:
                continue
                
            success, bbox = tracker_info['tracker'].update(frame)
            
            if success:
                tracker_info['bbox'] = bbox
                tracker_info['age'] += 1
                active_trackers[tracker_id] = tracker_info
            else:
                print(f"Tracker {tracker_id} lost target")
        
        self.trackers = active_trackers
    
    def draw_trackers(self, frame):
        """Draw all active trackers"""
        # Draw current selection
        if self.current_selection:
            x, y, w, h = self.current_selection
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(frame, "Selecting...", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw trackers
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for tracker_id, tracker_info in self.trackers.items():
            if not tracker_info['active']:
                continue
                
            bbox = tracker_info['bbox']
            x, y, w, h = map(int, bbox)
            color = colors[tracker_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw tracker info
            label = f"ID: {tracker_id} (Age: {tracker_info['age']})"
            cv2.putText(frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def clear_trackers(self):
        """Clear all trackers"""
        self.trackers = {}
        print("All trackers cleared!")

def main():
    print("=== KCF Tracking Demo ===")
    print("Instruksi:")
    print("- Klik dan drag untuk memilih objek yang ingin di-track")
    print("- Tekan 'c' untuk clear semua tracker")
    print("- Tekan 'q' untuk keluar")
    print("- Maximum 5 tracker bersamaan")
    print()
    
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"Error: Tidak dapat membuka kamera ID {CAMERA_ID}")
        return
    
    demo = SimpleKCFDemo()
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, demo.mouse_callback, None)
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Update trackers
        demo.update_trackers(frame)
        
        # Draw trackers
        demo.draw_trackers(frame)
        
        # Add statistics
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        stats_text = f"Trackers: {len(demo.trackers)} | FPS: {fps:.1f}"
        cv2.putText(frame, stats_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add instructions
        cv2.putText(frame, "Click & drag to select objects", (10, frame.shape[0] - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'c' to clear, 'q' to quit", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow(WINDOW_NAME, frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            demo.clear_trackers()
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()