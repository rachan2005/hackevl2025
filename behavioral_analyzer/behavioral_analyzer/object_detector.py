"""
Object Detection Module using YOLO.

This module provides real-time object detection capabilities using YOLOv8
from the Ultralytics library, integrated with the behavioral analysis system.
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
import threading
import uuid

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: YOLO not available. Install ultralytics to enable object detection.")


class ObjectDetector:
    """
    YOLO-based object detector for real-time object detection and tracking.
    
    Features:
    - Real-time object detection using YOLOv8
    - Object tracking and counting
    - Confidence filtering
    - Customizable detection classes
    - Performance monitoring
    """
    
    def __init__(self, model_size: str = "n", confidence_threshold: float = 0.5, 
                 device: str = "cpu", max_detections: int = 100):
        """
        Initialize the YOLO object detector.
        
        Args:
            model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
            confidence_threshold: Minimum confidence for detections
            device: Device to run inference on ('cpu', 'cuda', 'mps')
            max_detections: Maximum number of detections per frame
        """
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.max_detections = max_detections
        
        # Initialize model
        self.model = None
        self.model_loaded = False
        
        # Detection results
        self.current_detections = []
        self.detection_history = deque(maxlen=100)
        
        # Performance tracking
        self.inference_times = deque(maxlen=50)
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        
        # Object tracking
        self.object_counts = {}
        self.tracked_objects = {}
        self.object_enter_times = {}
        
        # Person tracking
        self.tracked_persons = {}
        self.person_tracking_history = deque(maxlen=100)
        self.main_person_id = None
        self.main_person_confidence = 0.0
        self.person_tracking_threshold = 0.7
        self.person_size_threshold = 0.1  # Minimum person size as fraction of frame
        
        # Detection statistics
        self.total_detections = 0
        self.detection_stats = {
            'total_frames': 0,
            'frames_with_detections': 0,
            'average_objects_per_frame': 0.0,
            'most_common_objects': {}
        }
        
        # Threading
        self.detection_thread = None
        self.detection_active = False
        self.frame_queue = deque(maxlen=5)
        self.result_queue = deque(maxlen=5)
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the YOLO model."""
        if not YOLO_AVAILABLE:
            print("YOLO not available. Object detection disabled.")
            return
        
        try:
            model_name = f"yolov8{self.model_size}.pt"
            print(f"Loading YOLO model: {model_name}")
            self.model = YOLO(model_name)
            self.model_loaded = True
            print(f"YOLO model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model_loaded = False
    
    def start_detection_thread(self):
        """Start the detection thread for asynchronous processing."""
        if not self.model_loaded:
            return False
        
        self.detection_active = True
        self.detection_thread = threading.Thread(target=self._detection_worker, daemon=True)
        self.detection_thread.start()
        return True
    
    def stop_detection_thread(self):
        """Stop the detection thread."""
        self.detection_active = False
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=1.0)
    
    def _detection_worker(self):
        """Worker thread for object detection."""
        while self.detection_active:
            if self.frame_queue:
                frame = self.frame_queue.popleft()
                detections = self._detect_objects(frame)
                self.result_queue.append(detections)
            else:
                time.sleep(0.01)  # Small delay to prevent busy waiting
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in the given frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of detection dictionaries
        """
        if not self.model_loaded:
            return []
        
        # Add frame to queue for processing
        self.frame_queue.append(frame.copy())
        
        # Get latest results
        if self.result_queue:
            return self.result_queue.popleft()
        
        # Fallback to synchronous detection
        return self._detect_objects(frame)
    
    def _detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Perform object detection on a frame."""
        start_time = time.time()
        
        try:
            # Run YOLO inference
            results = self.model(frame, conf=self.confidence_threshold, 
                               max_det=self.max_detections, device=self.device, verbose=False)
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        # Extract detection information
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.model.names[class_id]
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'class_name': class_name,
                            'timestamp': time.time(),
                            'area': (int(x2) - int(x1)) * (int(y2) - int(y1))
                        }
                        
                        detections.append(detection)
            
            # Update statistics
            self._update_detection_stats(detections)
            
            # Track persons
            self._track_persons(detections, frame.shape[:2])
            
            # Store current detections
            self.current_detections = detections
            self.detection_history.append(detections)
            
            # Update performance metrics
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self._update_fps()
            
            return detections
            
        except Exception as e:
            print(f"Error in object detection: {e}")
            return []
    
    def _update_detection_stats(self, detections: List[Dict[str, Any]]):
        """Update detection statistics."""
        self.detection_stats['total_frames'] += 1
        
        if detections:
            self.detection_stats['frames_with_detections'] += 1
            self.total_detections += len(detections)
            
            # Update object counts
            for detection in detections:
                class_name = detection['class_name']
                self.object_counts[class_name] = self.object_counts.get(class_name, 0) + 1
                
                # Update most common objects
                if class_name not in self.detection_stats['most_common_objects']:
                    self.detection_stats['most_common_objects'][class_name] = 0
                self.detection_stats['most_common_objects'][class_name] += 1
        
        # Calculate average objects per frame
        if self.detection_stats['total_frames'] > 0:
            self.detection_stats['average_objects_per_frame'] = (
                self.total_detections / self.detection_stats['total_frames']
            )
    
    def _update_fps(self):
        """Update FPS counter."""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """Get a summary of detection statistics."""
        return {
            'total_detections': self.total_detections,
            'current_detections': len(self.current_detections),
            'fps': self.current_fps,
            'average_inference_time': np.mean(self.inference_times) if self.inference_times else 0.0,
            'object_counts': self.object_counts.copy(),
            'detection_stats': self.detection_stats.copy(),
            'model_loaded': self.model_loaded,
            'model_size': self.model_size,
            'confidence_threshold': self.confidence_threshold
        }
    
    def get_objects_in_region(self, region: Tuple[int, int, int, int]) -> List[Dict[str, Any]]:
        """
        Get objects detected in a specific region.
        
        Args:
            region: (x1, y1, x2, y2) defining the region
            
        Returns:
            List of detections in the region
        """
        x1, y1, x2, y2 = region
        objects_in_region = []
        
        for detection in self.current_detections:
            det_x1, det_y1, det_x2, det_y2 = detection['bbox']
            
            # Check if detection overlaps with region
            if (det_x1 < x2 and det_x2 > x1 and 
                det_y1 < y2 and det_y2 > y1):
                objects_in_region.append(detection)
        
        return objects_in_region
    
    def get_objects_by_class(self, class_name: str) -> List[Dict[str, Any]]:
        """
        Get all detections of a specific class.
        
        Args:
            class_name: Name of the class to filter
            
        Returns:
            List of detections of the specified class
        """
        return [det for det in self.current_detections if det['class_name'] == class_name]
    
    def get_person_count(self) -> int:
        """
        Get the number of people detected in the current frame.
        
        Returns:
            Number of people detected
        """
        return len(self.get_objects_by_class('person'))
    
    def get_person_detections(self) -> List[Dict[str, Any]]:
        """
        Get all person detections in the current frame.
        
        Returns:
            List of person detections
        """
        return self.get_objects_by_class('person')
    
    def _track_persons(self, detections: List[Dict[str, Any]], frame_shape: Tuple[int, int]):
        """
        Track persons across frames and identify the main person.
        
        Args:
            detections: List of current detections
            frame_shape: (height, width) of the frame
        """
        current_persons = [det for det in detections if det['class_name'] == 'person']
        frame_height, frame_width = frame_shape
        
        # Calculate person areas and positions
        for person in current_persons:
            x1, y1, x2, y2 = person['bbox']
            person_area = (x2 - x1) * (y2 - y1)
            frame_area = frame_height * frame_width
            person_size_ratio = person_area / frame_area
            
            # Add size and position info
            person['size_ratio'] = person_size_ratio
            person['center_x'] = (x1 + x2) / 2
            person['center_y'] = (y1 + y2) / 2
            person['distance_from_center'] = np.sqrt(
                (person['center_x'] - frame_width/2)**2 + 
                (person['center_y'] - frame_height/2)**2
            )
        
        # Try to match with existing tracked persons
        matched_persons = set()
        for person in current_persons:
            best_match_id = None
            best_match_score = 0
            
            for person_id, tracked_person in self.tracked_persons.items():
                if person_id in matched_persons:
                    continue
                
                # Calculate matching score based on position and size
                position_score = self._calculate_position_similarity(person, tracked_person)
                size_score = self._calculate_size_similarity(person, tracked_person)
                confidence_score = person['confidence']
                
                total_score = (position_score * 0.4 + size_score * 0.3 + confidence_score * 0.3)
                
                if total_score > best_match_score and total_score > self.person_tracking_threshold:
                    best_match_score = total_score
                    best_match_id = person_id
            
            if best_match_id:
                # Update existing person
                self._update_tracked_person(best_match_id, person)
                matched_persons.add(best_match_id)
            else:
                # Create new person
                new_person_id = str(uuid.uuid4())
                self._create_tracked_person(new_person_id, person)
        
        # Remove persons that are no longer detected
        persons_to_remove = []
        for person_id in self.tracked_persons:
            if person_id not in matched_persons:
                self.tracked_persons[person_id]['frames_missing'] += 1
                if self.tracked_persons[person_id]['frames_missing'] > 10:  # Remove after 10 frames
                    persons_to_remove.append(person_id)
        
        for person_id in persons_to_remove:
            del self.tracked_persons[person_id]
        
        # Update main person
        self._update_main_person()
    
    def _calculate_position_similarity(self, person1: Dict, person2: Dict) -> float:
        """Calculate position similarity between two persons."""
        if 'center_x' not in person2 or 'center_y' not in person2:
            return 0.0
        
        distance = np.sqrt(
            (person1['center_x'] - person2['center_x'])**2 + 
            (person1['center_y'] - person2['center_y'])**2
        )
        
        # Normalize distance (assuming max distance is diagonal of frame)
        max_distance = np.sqrt(640**2 + 480**2)  # Approximate max distance
        similarity = max(0, 1 - (distance / max_distance))
        return similarity
    
    def _calculate_size_similarity(self, person1: Dict, person2: Dict) -> float:
        """Calculate size similarity between two persons."""
        if 'size_ratio' not in person2:
            return 0.0
        
        size_diff = abs(person1['size_ratio'] - person2['size_ratio'])
        similarity = max(0, 1 - (size_diff * 10))  # Scale factor for size difference
        return similarity
    
    def _create_tracked_person(self, person_id: str, person: Dict[str, Any]):
        """Create a new tracked person."""
        self.tracked_persons[person_id] = {
            'id': person_id,
            'bbox': person['bbox'],
            'confidence': person['confidence'],
            'center_x': person['center_x'],
            'center_y': person['center_y'],
            'size_ratio': person['size_ratio'],
            'distance_from_center': person['distance_from_center'],
            'first_seen': time.time(),
            'last_seen': time.time(),
            'frames_detected': 1,
            'frames_missing': 0,
            'total_confidence': person['confidence'],
            'is_main_person': False
        }
    
    def _update_tracked_person(self, person_id: str, person: Dict[str, Any]):
        """Update an existing tracked person."""
        tracked = self.tracked_persons[person_id]
        tracked['bbox'] = person['bbox']
        tracked['confidence'] = person['confidence']
        tracked['center_x'] = person['center_x']
        tracked['center_y'] = person['center_y']
        tracked['size_ratio'] = person['size_ratio']
        tracked['distance_from_center'] = person['distance_from_center']
        tracked['last_seen'] = time.time()
        tracked['frames_detected'] += 1
        tracked['frames_missing'] = 0
        tracked['total_confidence'] = (tracked['total_confidence'] * (tracked['frames_detected'] - 1) + person['confidence']) / tracked['frames_detected']
    
    def _update_main_person(self):
        """Update the main person based on tracking data."""
        if not self.tracked_persons:
            self.main_person_id = None
            self.main_person_confidence = 0.0
            return
        
        # Score each person based on multiple factors
        person_scores = {}
        for person_id, person in self.tracked_persons.items():
            # Factors for main person selection:
            # 1. Size (larger person is more likely to be main)
            # 2. Position (center of frame is more likely to be main)
            # 3. Confidence (higher confidence is better)
            # 4. Duration (longer presence is more stable)
            # 5. Consistency (more consistent detection)
            
            size_score = person['size_ratio'] * 10  # Scale up size ratio
            position_score = max(0, 1 - (person['distance_from_center'] / 400))  # Normalize distance
            confidence_score = person['total_confidence']
            duration_score = min(1.0, (time.time() - person['first_seen']) / 30)  # 30 seconds max
            consistency_score = person['frames_detected'] / max(1, person['frames_detected'] + person['frames_missing'])
            
            total_score = (
                size_score * 0.3 +
                position_score * 0.25 +
                confidence_score * 0.2 +
                duration_score * 0.15 +
                consistency_score * 0.1
            )
            
            person_scores[person_id] = total_score
        
        # Select person with highest score
        if person_scores:
            best_person_id = max(person_scores, key=person_scores.get)
            best_score = person_scores[best_person_id]
            
            # Only update main person if score is significantly better or current main person is gone
            if (self.main_person_id is None or 
                best_score > self.main_person_confidence + 0.1 or 
                self.main_person_id not in self.tracked_persons):
                
                # Clear previous main person
                if self.main_person_id and self.main_person_id in self.tracked_persons:
                    self.tracked_persons[self.main_person_id]['is_main_person'] = False
                
                # Set new main person
                self.main_person_id = best_person_id
                self.main_person_confidence = best_score
                self.tracked_persons[best_person_id]['is_main_person'] = True
    
    def get_main_person(self) -> Optional[Dict[str, Any]]:
        """
        Get the main person detection.
        
        Returns:
            Main person detection or None if no main person
        """
        if self.main_person_id and self.main_person_id in self.tracked_persons:
            return self.tracked_persons[self.main_person_id]
        return None
    
    def get_person_tracking_info(self) -> Dict[str, Any]:
        """
        Get comprehensive person tracking information.
        
        Returns:
            Dictionary with person tracking data
        """
        return {
            'total_persons': len(self.tracked_persons),
            'main_person_id': self.main_person_id,
            'main_person_confidence': self.main_person_confidence,
            'tracked_persons': list(self.tracked_persons.values()),
            'person_detections': self.get_person_detections()
        }
    
    def draw_detections(self, frame: np.ndarray, show_confidence: bool = True, 
                       show_class: bool = True, thickness: int = 2) -> np.ndarray:
        """
        Draw detection bounding boxes on the frame.
        
        Args:
            frame: Input frame
            show_confidence: Whether to show confidence scores
            show_class: Whether to show class names
            thickness: Line thickness for bounding boxes
            
        Returns:
            Frame with drawn detections
        """
        if not self.current_detections:
            return frame
        
        frame_copy = frame.copy()
        
        for detection in self.current_detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Choose color based on class and main person status
            if class_name == 'person':
                # Check if this is the main person
                main_person = self.get_main_person()
                if main_person and self._is_same_person(detection, main_person):
                    color = (0, 255, 0)  # Green for main person
                    thickness = max(thickness, 3)  # Thicker border for main person
                else:
                    color = (0, 255, 255)  # Yellow for other persons
            else:
                color = self._get_class_color(class_name)
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label
            label_parts = []
            if show_class:
                label_parts.append(class_name)
            if show_confidence:
                label_parts.append(f"{confidence:.2f}")
            
            if label_parts:
                label = " ".join(label_parts)
                
                # Get label size
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                # Draw label background
                cv2.rectangle(frame_copy, (x1, y1 - label_height - 10), 
                             (x1 + label_width, y1), color, -1)
                
                # Draw label text
                cv2.putText(frame_copy, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame_copy
    
    def _is_same_person(self, detection: Dict[str, Any], tracked_person: Dict[str, Any]) -> bool:
        """Check if a detection matches a tracked person."""
        if 'center_x' not in detection or 'center_y' not in detection:
            return False
        
        # Calculate distance between detection and tracked person
        distance = np.sqrt(
            (detection['center_x'] - tracked_person['center_x'])**2 + 
            (detection['center_y'] - tracked_person['center_y'])**2
        )
        
        # Consider it the same person if distance is small
        return distance < 50  # Adjust threshold as needed
    
    def _get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """Get a consistent color for a class."""
        # Generate a consistent color based on class name
        hash_value = hash(class_name) % 360
        hue = hash_value / 360.0
        
        # Convert HSV to BGR
        import colorsys
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        return (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_detection_thread()
        if self.model:
            del self.model
        print("Object detector cleaned up")
