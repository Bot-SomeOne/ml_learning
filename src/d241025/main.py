
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from datetime import datetime
import json
import sys

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON Encoder to handle numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class AdvancedVehicleCounter:
    def __init__(self, model_path='yolov8n.pt', line_coords=None):
        """
        Initialize vehicle counter
        
        Args:
            model_path: Path to YOLO model
            line_coords: Counting line coordinates [(x1, y1), (x2, y2)]
        """
        print("Initializing with Ultralytics 8.3.221")
        self.model = YOLO(model_path)
        
        # Vehicle classes according to COCO dataset
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck'
        }
        
        # Counting line
        self.line_coords = line_coords
        
        # Tracking data
        self.tracked_objects = {}
        self.counted_ids = set()
        
        # Count by type
        self.count_by_type = {name: 0 for name in self.vehicle_classes.values()}
        self.count_by_direction = {
            'up': {name: 0 for name in self.vehicle_classes.values()},
            'down': {name: 0 for name in self.vehicle_classes.values()}
        }
        self.total_count = 0
        
        # Colors for each vehicle type
        self.colors = {
            'car': (255, 0, 0),        # Blue
            'motorcycle': (0, 255, 0),  # Green
            'bus': (0, 165, 255),       # Orange
            'truck': (0, 0, 255)        # Red
        }
        
        # Mouse callback state
        self.drawing = False
        self.temp_line_coords = []
        
        # Statistics tracking
        self.frame_stats = []
        
    def set_line_coords(self, coords):
        """Set counting line coordinates"""
        self.line_coords = coords
        print(f"Counting line set: P1{coords[0]} -> P2{coords[1]}")
        
    def draw_line(self, frame):
        """Draw counting line"""
        if self.line_coords:
            (x1, y1), (x2, y2) = self.line_coords
            
            # Draw line with gradient effect
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 4)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw start and end points
            cv2.circle(frame, (x1, y1), 10, (0, 0, 255), -1)
            cv2.circle(frame, (x1, y1), 12, (255, 255, 255), 2)
            cv2.circle(frame, (x2, y2), 10, (0, 0, 255), -1)
            cv2.circle(frame, (x2, y2), 12, (255, 255, 255), 2)
            
            # Display coordinates with background
            self._draw_text_with_background(frame, f"P1({x1},{y1})", (x1 + 15, y1 - 10))
            self._draw_text_with_background(frame, f"P2({x2},{y2})", (x2 + 15, y2 - 10))
    
    def _draw_text_with_background(self, frame, text, pos, font_scale=0.5, thickness=1):
        """Draw text with background for better readability"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Background
        cv2.rectangle(frame, 
                     (pos[0] - 2, pos[1] - text_h - 2),
                     (pos[0] + text_w + 2, pos[1] + baseline + 2),
                     (0, 0, 0), -1)
        
        # Text
        cv2.putText(frame, text, pos, font, font_scale, (255, 255, 255), thickness)
    
    def get_center(self, box):
        """Get center coordinates of bounding box"""
        x1, y1, x2, y2 = box
        return int((x1 + x2) / 2), int((y1 + y2) / 2)
    
    def get_bottom_center(self, box):
        """Get bottom center of bounding box (more accurate for vehicles)"""
        x1, y1, x2, y2 = box
        return int((x1 + x2) / 2), int(y2)
    
    def ccw(self, A, B, C):
        """Check counter-clockwise orientation of three points"""
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    def line_intersect(self, A, B, C, D):
        """
        Check if line segment AB intersects line segment CD
        """
        return self.ccw(A, C, D) != self.ccw(B, C, D) and self.ccw(A, B, C) != self.ccw(A, B, D)
    
    def get_direction(self, prev_pos, curr_pos, line_p1, line_p2):
        """
        Determine movement direction (up/down or left/right)
        """
        # Line vector
        line_vec = (line_p2[0] - line_p1[0], line_p2[1] - line_p1[1])
        # Movement vector
        move_vec = (curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])
        
        # Cross product to determine direction
        cross = line_vec[0] * move_vec[1] - line_vec[1] * move_vec[0]
        
        return 'down' if cross > 0 else 'up'
    
    def process_frame(self, frame):
        """Process each frame with Ultralytics 8.3.221"""
        height, width = frame.shape[:2]
        
        # Set default counting line if not set
        if self.line_coords is None:
            self.line_coords = [(0, height // 2), (width, height // 2)]
        
        # Detect and track with ByteTrack
        results = self.model.track(
            frame,
            persist=True,
            classes=list(self.vehicle_classes.keys()),
            conf=0.25,
            iou=0.45,
            tracker="bytetrack.yaml",
            verbose=False,
            imgsz=640,
        )
        
        current_frame_vehicles = []
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for box, track_id, cls, conf in zip(boxes, track_ids, classes, confidences):
                x1, y1, x2, y2 = box
                cx, cy = self.get_center(box)
                bottom_cx, bottom_cy = self.get_bottom_center(box)
                
                # Convert numpy types to Python native types
                track_id = int(track_id)
                cls = int(cls)
                conf = float(conf)
                
                # Get vehicle type
                vehicle_type = self.vehicle_classes.get(cls, 'unknown')
                color = self.colors.get(vehicle_type, (255, 255, 255))
                
                current_frame_vehicles.append({
                    'id': track_id,
                    'type': vehicle_type,
                    'position': (int(cx), int(cy)),
                    'confidence': conf
                })
                
                # Draw bounding box
                thickness = 3 if conf > 0.5 else 2
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                
                # Draw tracking point (using bottom center)
                cv2.circle(frame, (bottom_cx, bottom_cy), 6, (0, 255, 255), -1)
                cv2.circle(frame, (bottom_cx, bottom_cy), 8, color, 2)
                
                # Tracking and counting
                if track_id not in self.tracked_objects:
                    self.tracked_objects[track_id] = {
                        'positions': [(bottom_cx, bottom_cy)],
                        'counted': False,
                        'type': vehicle_type,
                        'first_seen': datetime.now().isoformat(),
                        'direction': None
                    }
                else:
                    prev_pos = self.tracked_objects[track_id]['positions'][-1]
                    curr_pos = (bottom_cx, bottom_cy)
                    self.tracked_objects[track_id]['positions'].append(curr_pos)
                    
                    # Limit positions storage
                    if len(self.tracked_objects[track_id]['positions']) > 30:
                        self.tracked_objects[track_id]['positions'].pop(0)
                    
                    # Check if trajectory crosses counting line
                    if not self.tracked_objects[track_id]['counted']:
                        line_p1, line_p2 = self.line_coords
                        
                        if self.line_intersect(prev_pos, curr_pos, line_p1, line_p2):
                            # Determine direction
                            direction = self.get_direction(prev_pos, curr_pos, line_p1, line_p2)
                            
                            # Count
                            self.total_count += 1
                            self.count_by_type[vehicle_type] += 1
                            self.count_by_direction[direction][vehicle_type] += 1
                            self.tracked_objects[track_id]['counted'] = True
                            self.tracked_objects[track_id]['direction'] = direction
                            self.counted_ids.add(track_id)
                            
                            # Draw effect when counting
                            cv2.circle(frame, (bottom_cx, bottom_cy), 40, (0, 255, 0), 4)
                            cv2.circle(frame, (bottom_cx, bottom_cy), 35, (0, 255, 255), 2)
                            
                            # Log event
                            print(f"Counted: {vehicle_type.upper()} #{track_id} - {direction.upper()} - Conf: {conf:.2f}")
                
                # Display information
                is_counted = self.tracked_objects[track_id]['counted']
                direction = self.tracked_objects[track_id].get('direction', '')
                
                if is_counted:
                    arrow = "UP" if direction == 'up' else "DOWN"
                    label = f"{vehicle_type} #{track_id} {arrow}"
                    label_color = (0, 255, 0)
                else:
                    label = f"{vehicle_type} #{track_id}"
                    label_color = (255, 255, 255)
                
                # Background for label
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, 
                            (int(x1), int(y1) - label_h - 15),
                            (int(x1) + label_w + 10, int(y1)),
                            color, -1)
                
                cv2.putText(frame, label, (int(x1) + 5, int(y1) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)
                
                # Confidence score
                conf_text = f"{conf:.0%}"
                cv2.putText(frame, conf_text, (int(x2) - 50, int(y2) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                # Draw trajectory with gradient
                if len(self.tracked_objects[track_id]['positions']) > 1:
                    points = self.tracked_objects[track_id]['positions']
                    for i in range(len(points) - 1):
                        # Calculate alpha for gradient effect
                        alpha = int(255 * (i + 1) / len(points))
                        thickness = max(1, int(3 * (i + 1) / len(points)))
                        
                        # Create color with alpha
                        point_color = tuple([int(c * alpha / 255) for c in color])
                        cv2.line(frame, points[i], points[i + 1], point_color, thickness)
        
        # Draw counting line
        self.draw_line(frame)
        
        # Draw statistics
        self.draw_statistics(frame)
        
        # Save stats for this frame (convert to native Python types)
        self.frame_stats.append({
            'timestamp': datetime.now().isoformat(),
            'total': int(self.total_count),
            'by_type': {k: int(v) for k, v in self.count_by_type.items()},
            'vehicles_in_frame': len(current_frame_vehicles)
        })
        
        return frame
    
    def draw_statistics(self, frame):
        """Draw statistics panel"""
        height, width = frame.shape[:2]
        
        # Panel 1: Main statistics (left)
        panel_width = 450
        panel_height = 320
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Border
        cv2.rectangle(frame, (10, 10), (panel_width, panel_height), (0, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "VEHICLE COUNTER", (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.line(frame, (20, 55), (panel_width - 20, 55), (0, 255, 255), 2)
        
        y_offset = 90
        
        # Statistics by vehicle type with icon and bar
        for vehicle_type, count in self.count_by_type.items():
            color = self.colors[vehicle_type]
            
            # Color icon
            cv2.rectangle(frame, (20, y_offset - 20), (45, y_offset), color, -1)
            cv2.rectangle(frame, (20, y_offset - 20), (45, y_offset), (255, 255, 255), 2)
            
            # Name and count
            text = f"{vehicle_type.capitalize()}: {count}"
            cv2.putText(frame, text, (55, y_offset - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Progress bar
            bar_width = int((count / max(self.total_count, 1)) * 150)
            cv2.rectangle(frame, (270, y_offset - 18), (270 + bar_width, y_offset - 2),
                         color, -1)
            cv2.rectangle(frame, (270, y_offset - 18), (420, y_offset - 2),
                         (100, 100, 100), 1)
            
            y_offset += 45
        
        # Separator line
        cv2.line(frame, (20, y_offset), (panel_width - 20, y_offset), (100, 100, 100), 2)
        y_offset += 30
        
        # Total with highlight
        total_text = f"TOTAL: {self.total_count}"
        cv2.putText(frame, total_text, (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
        
        # Panel 2: Direction statistics (top right)
        panel2_x = width - 300
        panel2_y = 10
        panel2_w = 290
        panel2_h = 180
        
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (panel2_x, panel2_y), 
                     (panel2_x + panel2_w, panel2_y + panel2_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (panel2_x, panel2_y), 
                     (panel2_x + panel2_w, panel2_y + panel2_h), (255, 165, 0), 2)
        
        cv2.putText(frame, "DIRECTION", (panel2_x + 10, panel2_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        
        y_dir = panel2_y + 70
        
        # Up direction
        up_total = sum(self.count_by_direction['up'].values())
        cv2.putText(frame, f"UP: {up_total}", (panel2_x + 20, y_dir),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Down direction  
        down_total = sum(self.count_by_direction['down'].values())
        cv2.putText(frame, f"DOWN: {down_total}", (panel2_x + 20, y_dir + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 255), 2)
        
        # Time at bottom
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, current_time, (width - 280, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def reset(self):
        """Reset all counters"""
        self.tracked_objects.clear()
        self.counted_ids.clear()
        self.count_by_type = {name: 0 for name in self.vehicle_classes.values()}
        self.count_by_direction = {
            'up': {name: 0 for name in self.vehicle_classes.values()},
            'down': {name: 0 for name in self.vehicle_classes.values()}
        }
        self.total_count = 0
        self.frame_stats = []
        print("Counter reset!")
    
    def export_statistics(self, filename='vehicle_stats.json'):
        """Export statistics to JSON file - Fixed JSON serialization"""
        # Convert counted_ids (set of int) to list
        counted_ids_list = [int(id) for id in self.counted_ids]
        
        stats = {
            'total_count': int(self.total_count),
            'count_by_type': {k: int(v) for k, v in self.count_by_type.items()},
            'count_by_direction': {
                direction: {k: int(v) for k, v in counts.items()}
                for direction, counts in self.count_by_direction.items()
            },
            'counted_ids': counted_ids_list,
            'line_coords': self.line_coords if self.line_coords else None,
            'timestamp': datetime.now().isoformat(),
            'vehicle_classes': self.vehicle_classes
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            print(f"Statistics exported to: {filename}")
            return True
        except Exception as e:
            print(f"Export error: {e}")
            return False

def mouse_callback(event, x, y, flags, param):
    """Mouse callback for drawing counting line"""
    counter, frame_copy = param
    
    if event == cv2.EVENT_LBUTTONDOWN:
        counter.drawing = True
        counter.temp_line_coords = [(x, y)]
        print(f"Point 1: ({x}, {y})")
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if counter.drawing and len(counter.temp_line_coords) > 0:
            temp_frame = frame_copy.copy()
            cv2.line(temp_frame, counter.temp_line_coords[0], (x, y), (0, 255, 0), 3)
            cv2.circle(temp_frame, counter.temp_line_coords[0], 8, (0, 0, 255), -1)
            cv2.circle(temp_frame, (x, y), 8, (0, 0, 255), -1)
            cv2.imshow('Vehicle Counter', temp_frame)
    
    elif event == cv2.EVENT_LBUTTONUP:
        counter.drawing = False
        if len(counter.temp_line_coords) > 0:
            counter.temp_line_coords.append((x, y))
            counter.set_line_coords(counter.temp_line_coords)
            print(f"Point 2: ({x}, {y})")
            counter.temp_line_coords = []

def main(video_path, output_path=None, line_coords=None, export_stats=True):
    """
    Main function to process video
    
    Args:
        video_path: Input video path (or 0 for webcam)
        output_path: Output video path (optional)
        line_coords: Counting line coordinates [(x1, y1), (x2, y2)] (optional)
        export_stats: Export statistics to JSON (default: True)
    """
    print("="*60)
    print("VEHICLE COUNTER - Ultralytics 8.3.221")
    print("="*60)
    
    # Initialize counter
    counter = AdvancedVehicleCounter(model_path='yolov8n.pt', line_coords=line_coords)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Cannot open video!")
        return
    
    # Get video information
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps}fps ({total_frames} frames)")
    
    # Initialize video writer with better codec
    writer = None
    if output_path:
        # Try different codecs for better compatibility
        fourcc_options = [
            cv2.VideoWriter_fourcc(*'mp4v'),  # MPEG-4
            cv2.VideoWriter_fourcc(*'avc1'),  # H.264
            cv2.VideoWriter_fourcc(*'X264'),  # H.264
            cv2.VideoWriter_fourcc(*'XVID'),  # Xvid
        ]
        
        for fourcc in fourcc_options:
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if writer.isOpened():
                print(f"Video writer initialized successfully")
                print(f"Saving to: {output_path}")
                break
        
        if not writer or not writer.isOpened():
            print("Warning: Could not initialize video writer. Video will not be saved.")
            writer = None
    
    print("\n" + "="*60)
    print("KEYBOARD SHORTCUTS:")
    print("="*60)
    print("  q / ESC  - Quit")
    print("  r        - Reset counter")
    print("  p        - Pause/Resume")
    print("  l        - Draw new counting line")
    print("  s        - Export statistics")
    print("  Mouse    - Click 2 points to draw line")
    print("="*60 + "\n")
    
    frame_count = 0
    paused = False
    frame_copy = None
    
    cv2.namedWindow('Vehicle Counter', cv2.WINDOW_NORMAL)
    
    import time
    start_time = time.time()
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                
                if not ret:
                    print("\nVideo ended")
                    break
                
                frame_count += 1
                frame_copy = frame.copy()
                
                # Process frame
                processed_frame = counter.process_frame(frame)
                
                # Calculate FPS
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                
                # Display information
                info_text = f"Frame: {frame_count}/{total_frames} | FPS: {current_fps:.1f}"
                cv2.putText(processed_frame, info_text, (width - 350, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                status = "PAUSED" if paused else "RUNNING"
                status_color = (0, 165, 255) if paused else (0, 255, 0)
                cv2.putText(processed_frame, status, (width - 350, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                
                # Progress bar
                progress = frame_count / total_frames if total_frames > 0 else 0
                bar_width = int(width * 0.3)
                bar_filled = int(bar_width * progress)
                bar_x = width - bar_width - 20
                bar_y = height - 40
                
                cv2.rectangle(processed_frame, (bar_x, bar_y), 
                             (bar_x + bar_width, bar_y + 10), (100, 100, 100), -1)
                cv2.rectangle(processed_frame, (bar_x, bar_y),
                             (bar_x + bar_filled, bar_y + 10), (0, 255, 0), -1)
                cv2.putText(processed_frame, f"{progress*100:.1f}%", 
                           (bar_x + bar_width + 10, bar_y + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Set mouse callback
                cv2.setMouseCallback('Vehicle Counter', mouse_callback, (counter, frame_copy))
                
                # Display frame
                cv2.imshow('Vehicle Counter', processed_frame)
                
                # Save video
                if writer and writer.isOpened():
                    writer.write(processed_frame)
            else:
                cv2.imshow('Vehicle Counter', processed_frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:
                print("\nExiting...")
                break
            elif key == ord('r'):
                counter.reset()
            elif key == ord('p'):
                paused = not paused
                print("PAUSED" if paused else "RESUMED")
            elif key == ord('l'):
                print("Draw counting line: Click 2 points...")
                counter.temp_line_coords = []
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                counter.export_statistics(f'stats_{timestamp}.json')
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        cap.release()
        if writer and writer.isOpened():
            writer.release()
            print(f"Video saved successfully: {output_path}")
        cv2.destroyAllWindows()
        
        # Export final statistics
        if export_stats:
            counter.export_statistics()
        
        # Print results
        elapsed = time.time() - start_time
        current_fps = frame_count / elapsed if elapsed > 0 else 0
        
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(f"Processing time: {elapsed:.2f}s")
        print(f"Frames: {frame_count} ({current_fps:.1f} FPS)")
        print("\nSTATISTICS BY TYPE:")
        print("-"*60)
        for vtype, count in counter.count_by_type.items():
            print(f"  {vtype.capitalize():<12}: {count:>5} vehicles")
        print("-"*60)
        print(f"  {'TOTAL':<12}: {counter.total_count:>5} vehicles")
        
        print("\nSTATISTICS BY DIRECTION:")
        print("-"*60)
        up_total = sum(counter.count_by_direction['up'].values())
        down_total = sum(counter.count_by_direction['down'].values())
        print(f"  Up           : {up_total:>5} vehicles")
        print(f"  Down         : {down_total:>5} vehicles")
        print("="*60)

if __name__ == "__main__":
    # ===== CONFIGURATION =====
    
    video_path = "traffic.mp4"
    output_path = "output_counted.mp4"
    
    # Option 1: Default horizontal line in the middle
    main(video_path, output_path)
    
    # Option 2: Custom coordinates
    # main(video_path, output_path, line_coords=[(100, 400), (1180, 400)])
    
    # Option 3: Webcam
    # main(0, None)
    
    # Option 4: No video output, just display
    # main(video_path, None)