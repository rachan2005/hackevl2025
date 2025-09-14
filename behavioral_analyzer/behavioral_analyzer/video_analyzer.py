"""
Video-based behavioral analysis module.

This module provides comprehensive video analysis capabilities including:
- Facial expression recognition
- Eye tracking and blink detection
- Attention analysis
- Posture analysis
- Movement analysis
- Fatigue detection
"""

import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace
import time
import threading
from collections import deque
import math
from scipy.signal import savgol_filter
import warnings

from .config import VideoConfig
from .utils import (
    PerformanceTracker, DataCollector, FrameProcessor, 
    MathUtils, ColorUtils, ensure_directory_exists
)
from .object_detector import ObjectDetector

warnings.filterwarnings("ignore")


class VideoAnalyzer:
    """Comprehensive video-based behavioral analysis."""
    
    def __init__(self, config: VideoConfig, data_collector: DataCollector = None):
        self.config = config
        self.data_collector = data_collector
        self.performance_tracker = PerformanceTracker()
        
        # Initialize MediaPipe components
        self._initialize_mediapipe()
        
        # Initialize analysis parameters
        self._initialize_analysis_parameters()
        
        # Initialize emotion detection if enabled
        if config.enable_emotion:
            self._initialize_emotion_detection()
        
        # Initialize object detection
        if config.enable_object_detection:
            self._initialize_object_detection()
        
        # Initialize session data
        self.session_start_time = time.time()
        
        print("Video analyzer initialized successfully")
    
    def _initialize_mediapipe(self):
        """Initialize MediaPipe face mesh and pose detection."""
        # Face mesh for facial landmarks
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_landmarks=True
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Pose detection for body analysis
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def _initialize_analysis_parameters(self):
        """Initialize analysis parameters and data structures."""
        # Emotion analysis
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.current_emotion = "Unknown"
        self.emotion_scores = {emotion: 0.0 for emotion in self.emotion_labels}
        self.emotion_detected = False
        self.emotion_result = None
        
        # Blink detection
        self._initialize_blink_detection()
        
        # Attention analysis
        self.attention_state = "Unknown"
        self.attention_history = deque(maxlen=60)
        
        # Posture analysis
        self.posture_state = "Unknown"
        self.posture_history = deque(maxlen=30)
        
        # Movement analysis
        self.movement_level = "Unknown"
        self.movement_history = deque(maxlen=30)
        self.movement_magnitude = deque(maxlen=60)
        
        # Fatigue detection
        self.fatigue_level = "Normal"
        self.drowsiness_frames = 0
        self.drowsiness_alerts = 0
    
    def _initialize_blink_detection(self):
        """Initialize blink detection parameters."""
        # Blink tracking
        self.blink_count = 0
        self.total_blink_count = 0
        self.last_reset_time = time.time()
        self.last_blink_time = time.time()
        self.eye_closed = False
        
        # Frame counters
        self.eye_closed_frames = 0
        self.eye_open_frames = 0
        
        # EAR (Eye Aspect Ratio) tracking
        self.ear_values = deque(maxlen=300)
        self.ear_baseline = deque(maxlen=50)
        self.ear_closed = deque(maxlen=20)
        self.left_ear_values = deque(maxlen=100)
        self.right_ear_values = deque(maxlen=100)
        
        # Calibration
        self.min_ear = 1.0
        self.max_ear = 0.0
        self.adaptive_threshold = 0.2
        self.is_calibrated = False
        self.calibration_counter = 0
        self.recalibration_interval = 100
        
        # Eye landmark indices
        self.left_eye_landmarks = {
            'top': 159, 'bottom': 145, 'left': 33, 'right': 133,
            'top_inner': 158, 'top_outer': 160, 'bottom_inner': 144, 'bottom_outer': 153
        }
        self.right_eye_landmarks = {
            'top': 386, 'bottom': 374, 'left': 362, 'right': 263,
            'top_inner': 385, 'top_outer': 387, 'bottom_inner': 373, 'bottom_outer': 380
        }
        
        # Blink analysis
        self.blink_duration = deque(maxlen=30)
        self.blink_intervals = deque(maxlen=30)
        self.blink_timestamps = deque(maxlen=100)
        self.ear_moving_avg = deque(maxlen=5)
    
    def _initialize_emotion_detection(self):
        """Initialize emotion detection system."""
        self.last_emotion_time = time.time()
        self.emotion_queue = deque(maxlen=5)
        self.emotion_result = None
        self.emotion_thread_active = True
        
        # Start emotion processing thread
        self.emotion_thread = threading.Thread(target=self._process_emotions)
        self.emotion_thread.daemon = True
        self.emotion_thread.start()
    
    def _initialize_object_detection(self):
        """Initialize YOLO object detection."""
        try:
            self.object_detector = ObjectDetector(
                model_size=self.config.yolo_model_size,
                confidence_threshold=self.config.yolo_confidence_threshold,
                device=self.config.yolo_device,
                max_detections=self.config.yolo_max_detections
            )
            
            # Start detection thread
            if self.object_detector.start_detection_thread():
                print("Object detection initialized successfully")
            else:
                print("Object detection failed to initialize")
                self.object_detector = None
                
        except Exception as e:
            print(f"Error initializing object detection: {e}")
            self.object_detector = None
    
    def _process_emotions(self):
        """Background thread for emotion processing."""
        print("Emotion detection thread started")
        while self.emotion_thread_active:
            if len(self.emotion_queue) > 0:
                try:
                    frame = self.emotion_queue.popleft()
                    if frame is None or frame.size == 0:
                        continue
                    
                    if len(frame.shape) != 3 or frame.shape[2] != 3:
                        continue
                    
                    # Analyze emotion
                    emotion_analysis = DeepFace.analyze(
                        frame,
                        actions=['emotion'],
                        enforce_detection=False,
                        detector_backend=self.config.emotion_backend,
                        silent=True
                    )
                    
                    if isinstance(emotion_analysis, list) and len(emotion_analysis) > 0:
                        self.emotion_result = emotion_analysis[0]
                        self.emotion_detected = True
                        
                        # Normalize emotion scores
                        total_score = sum(self.emotion_result['emotion'].values())
                        if total_score > 0:
                            for emotion in self.emotion_labels:
                                raw_score = self.emotion_result['emotion'].get(emotion, 0)
                                self.emotion_scores[emotion] = min(1.0, raw_score / max(1.0, total_score))
                        
                        print(f"Emotion detected: {self.emotion_result['dominant_emotion']}")
                
                except Exception as e:
                    print(f"Emotion analysis error: {e}")
            
            time.sleep(0.05)
    
    def analyze_frame(self, frame: np.ndarray) -> np.ndarray:
        """Analyze a single frame and return annotated frame."""
        process_start = time.time()
        
        # Create display frame
        display_frame = frame.copy()
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with face mesh and pose
        face_results = self.face_mesh.process(rgb_frame)
        pose_results = self.pose.process(rgb_frame)
        
        ear = 0
        threshold = 0
        
        # Process face landmarks (focus on main person only)
        if face_results.multi_face_landmarks:
            # If we have a main person, only process faces within their bounding box
            main_person = getattr(self, 'main_person', None)
            
            for i, face_landmarks in enumerate(face_results.multi_face_landmarks):
                # Check if this face belongs to the main person
                if main_person and not self._is_face_in_main_person_region(face_landmarks, main_person):
                    continue  # Skip faces not belonging to main person
                
                # Draw landmarks if enabled
                if self.config.show_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image=display_frame,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
                
                # Emotion analysis
                if self.config.enable_emotion:
                    self._process_emotion_frame(frame, face_landmarks)
                
                # Blink detection
                if self.config.enable_blink_detection:
                    ear, threshold = self._detect_blinks(face_landmarks, display_frame)
                
                # Attention analysis
                if self.config.enable_attention_analysis:
                    self._analyze_attention(face_landmarks, display_frame)
                
                # Fatigue analysis
                if self.config.enable_fatigue_detection:
                    self._analyze_fatigue(ear, threshold)
                
                # Only process the first face that belongs to main person
                if main_person:
                    break
        
        # Process pose landmarks
        if pose_results.pose_landmarks:
            if self.config.show_landmarks:
                self.mp_drawing.draw_landmarks(
                    display_frame,
                    pose_results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
            
            # Posture analysis
            if self.config.enable_posture_analysis:
                self._analyze_posture(pose_results.pose_landmarks, display_frame)
            
            # Movement analysis
            if self.config.enable_movement_analysis:
                self._analyze_movement(pose_results.pose_landmarks, display_frame)
        
        # Update emotion from background thread
        if self.emotion_result:
            self.current_emotion = self.emotion_result['dominant_emotion']
            self.emotion_result = None
        
        # Perform object detection
        if hasattr(self, 'object_detector') and self.object_detector:
            detections = self.object_detector.detect_objects(frame)
            self.current_detections = detections
            
            # Get main person for focused analysis
            main_person = self.object_detector.get_main_person()
            self.main_person = main_person
            
            # Draw object detections if enabled
            if self.config.show_object_detections:
                display_frame = self.object_detector.draw_detections(
                    display_frame, 
                    show_confidence=self.config.show_detection_confidence,
                    show_class=self.config.show_detection_class
                )
        
        # Collect data for output
        if self.data_collector:
            self._collect_video_data(ear, threshold)
        
        # Update performance metrics
        process_end = time.time()
        process_time = process_end - process_start
        self.performance_tracker.update(process_time)
        
        # Display results
        self._display_results(display_frame, ear, threshold)
        
        return display_frame
    
    def _process_emotion_frame(self, frame: np.ndarray, face_landmarks):
        """Process frame for emotion detection."""
        current_time = time.time()
        if current_time - self.last_emotion_time > self.config.emotion_cooldown / 2:
            face_region = FrameProcessor.extract_face_region(frame, face_landmarks, expand_ratio=1.5)
            if face_region is not None:
                self.emotion_queue.append(face_region)
                self.last_emotion_time = current_time
    
    def _detect_blinks(self, face_landmarks, frame: np.ndarray) -> tuple:
        """Detect blinks using eye aspect ratio."""
        try:
            h, w, _ = frame.shape
            
            # Calculate EAR for both eyes
            left_ear, left_width = self._calculate_ear(face_landmarks, self.left_eye_landmarks, frame)
            right_ear, right_width = self._calculate_ear(face_landmarks, self.right_eye_landmarks, frame)
            
            # Store individual eye values
            self.left_ear_values.append(left_ear)
            self.right_ear_values.append(right_ear)
            
            # Average EAR
            ear = (left_ear + right_ear) / 2.0
            
            # Calculate dynamic threshold
            threshold = self._calculate_dynamic_threshold(ear)
            
            # Blink detection state machine
            self._update_blink_state(ear, threshold)
            
            # Recalibrate periodically
            self.calibration_counter += 1
            if self.calibration_counter >= self.recalibration_interval:
                self._recalibrate()
                self.calibration_counter = 0
            
            return ear, threshold
            
        except Exception as e:
            if self.config.debug_mode:
                print(f"Error in blink detection: {e}")
            return 0, 0
    
    def _calculate_ear(self, landmarks, eye_points: dict, frame: np.ndarray = None) -> tuple:
        """Calculate eye aspect ratio."""
        # Get eye landmark points
        top = landmarks.landmark[eye_points['top']]
        bottom = landmarks.landmark[eye_points['bottom']]
        left = landmarks.landmark[eye_points['left']]
        right = landmarks.landmark[eye_points['right']]
        
        # Calculate height and width
        height = abs(top.y - bottom.y)
        width = abs(right.x - left.x)
        
        # Calculate EAR
        ear = height / max(width, 0.001)
        
        return ear, width
    
    def _calculate_dynamic_threshold(self, ear: float) -> float:
        """Calculate dynamic threshold for blink detection."""
        self.ear_values.append(ear)
        
        if len(self.ear_values) > 15:
            recent_ears = list(self.ear_values)[-50:]
            
            # Apply smoothing
            if len(recent_ears) > 10:
                try:
                    window_length = min(len(recent_ears)-1, 7)
                    if window_length % 2 == 0:
                        window_length -= 1
                    if window_length >= 5:
                        recent_ears = savgol_filter(recent_ears, window_length, 2).tolist()
                except:
                    pass
            
            # Calculate percentiles
            sorted_ears = sorted(recent_ears)
            q10 = sorted_ears[len(sorted_ears)//10]
            q75 = sorted_ears[3*len(sorted_ears)//4]
            
            self.min_ear = q10 * 0.9
            self.max_ear = q75 * 1.1
            
            percentile_threshold = q10 + (q75 - q10) * 0.3
        else:
            percentile_threshold = self.adaptive_threshold
        
        # Baseline-based threshold
        if len(self.ear_baseline) > 5:
            baseline_mean = sum(self.ear_baseline) / len(self.ear_baseline)
            baseline_std = np.std(list(self.ear_baseline))
            baseline_threshold = baseline_mean - (2.0 * baseline_std)
            
            if not self.is_calibrated and len(self.ear_baseline) > 15:
                self.is_calibrated = True
        else:
            baseline_threshold = 0.2
        
        # Combine thresholds
        if self.is_calibrated:
            threshold = baseline_threshold * 0.7 + percentile_threshold * 0.3
        else:
            threshold = percentile_threshold * 0.7 + baseline_threshold * 0.3
        
        # Ensure reasonable threshold
        threshold = max(min(threshold, self.max_ear * 0.8), self.min_ear * 1.2)
        
        return threshold
    
    def _update_blink_state(self, ear: float, threshold: float):
        """Update enhanced blink detection state machine."""
        # Add to moving average
        self.ear_moving_avg.append(ear)
        ear_avg = sum(self.ear_moving_avg) / len(self.ear_moving_avg)
        
        # Enhanced drowsiness detection
        if ear_avg < threshold * 1.2 and ear_avg > threshold * 0.8:
            self.drowsiness_frames += 1
        else:
            self.drowsiness_frames = max(0, self.drowsiness_frames - 1)
        
        if self.drowsiness_frames > 20:
            self.drowsiness_alerts += 1
            self.drowsiness_frames = 0
        
        # Enhanced blink state machine with better validation
        if ear_avg < threshold:
            self.eye_closed_frames += 1
            self.eye_open_frames = 0
            self.ear_closed.append(ear)
            
            # Only register as closed if we have enough consecutive frames
            if (self.eye_closed_frames >= self.config.eye_ar_consec_frames_closed and 
                not self.eye_closed):
                self.eye_closed = True
                self.blink_start_time = time.time()
                
                # Debug output
                if self.config.debug_mode:
                    print(f"Eye closed: EAR={ear:.4f}, threshold={threshold:.4f}, frames={self.eye_closed_frames}")
        else:
            # Only add to baseline if definitely open (well above threshold)
            if ear > threshold * 1.2:
                self.ear_baseline.append(ear)
            
            self.eye_open_frames += 1
            
            # Complete blink detection
            if (self.eye_open_frames >= self.config.eye_ar_consec_frames_open and 
                self.eye_closed):
                self.eye_closed = False
                
                current_time = time.time()
                
                # Enhanced blink validation
                if current_time - self.last_blink_time > self.config.blink_cooldown:
                    # Calculate blink duration
                    blink_duration = current_time - self.blink_start_time
                    
                    # Validate blink duration (typical blinks are 100-400ms)
                    if 0.05 <= blink_duration <= 1.0:  # Valid blink duration
                        self.blink_count += 1
                        self.total_blink_count += 1
                        
                        # Store blink metrics
                        self.blink_duration.append(blink_duration)
                        
                        # Calculate interval since last blink
                        if self.blink_timestamps:
                            interval = current_time - self.blink_timestamps[-1]
                            self.blink_intervals.append(interval)
                        
                        # Store timestamp
                        self.blink_timestamps.append(current_time)
                        self.last_blink_time = current_time
                        
                        # Debug output
                        if self.config.debug_mode:
                            print(f"Blink detected #{self.total_blink_count}: "
                                  f"duration={blink_duration:.3f}s, "
                                  f"interval={interval:.3f}s" if self.blink_timestamps else "first blink")
                    else:
                        # Invalid blink duration - might be extended closure
                        if self.config.debug_mode:
                            print(f"Invalid blink duration: {blink_duration:.3f}s (ignored)")
            
            self.eye_closed_frames = 0
    
    def _recalibrate(self):
        """Recalibrate blink detection thresholds."""
        if len(self.ear_baseline) > 5 and len(self.ear_closed) > 2:
            open_mean = sum(self.ear_baseline) / len(self.ear_baseline)
            open_std = np.std(list(self.ear_baseline))
            closed_mean = sum(self.ear_closed) / len(self.ear_closed)
            closed_std = np.std(list(self.ear_closed))
            
            separation_ratio = abs(open_mean - closed_mean) / max((open_std + closed_std), 0.001)
            
            if separation_ratio > 1.0:
                weight = min(0.5, max(0.2, 1.0 / separation_ratio))
                self.adaptive_threshold = closed_mean + (open_mean - closed_mean) * weight
                
                if self.config.debug_mode:
                    print(f"Recalibrated threshold: {self.adaptive_threshold:.4f}")
            else:
                self.ear_closed = deque(maxlen=20)
    
    def _analyze_attention(self, face_landmarks, frame: np.ndarray):
        """Analyze attention based on face orientation."""
        try:
            h, w, _ = frame.shape
            
            # Extract key landmarks
            nose_tip = face_landmarks.landmark[4]
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            forehead = face_landmarks.landmark[10]
            chin = face_landmarks.landmark[152]
            
            # Calculate face orientation
            horizontal_vector = (right_eye.x - left_eye.x, right_eye.y - left_eye.y)
            vertical_vector = (chin.y - forehead.y, forehead.x - chin.x)
            
            # Normalize vectors
            h_magnitude = math.sqrt(horizontal_vector[0]**2 + horizontal_vector[1]**2)
            horizontal_vector = (horizontal_vector[0]/max(h_magnitude, 0.001), 
                                horizontal_vector[1]/max(h_magnitude, 0.001))
            
            v_magnitude = math.sqrt(vertical_vector[0]**2 + vertical_vector[1]**2)
            vertical_vector = (vertical_vector[0]/max(v_magnitude, 0.001), 
                              vertical_vector[1]/max(v_magnitude, 0.001))
            
            # Calculate angles
            horizontal_angle = math.degrees(math.atan2(horizontal_vector[1], horizontal_vector[0]))
            horizontal_angle = (horizontal_angle + 90) % 360
            if horizontal_angle > 180:
                horizontal_angle -= 360
            
            vertical_angle = math.degrees(math.atan2(vertical_vector[1], vertical_vector[0]))
            
            # Position-based factor
            nose_x = int(nose_tip.x * w)
            frame_center_x = w // 2
            distance_from_center = abs(nose_x - frame_center_x) / (w * 0.5)
            
            # Classify attention
            if abs(horizontal_angle) < 25 and abs(vertical_angle) < 25:
                new_state = "Focused"
            elif (abs(horizontal_angle) < 45 and abs(vertical_angle) < 30) or distance_from_center < 0.35:
                new_state = "Partially Attentive"
            else:
                new_state = "Distracted"
            
            self.attention_state = new_state
            self.attention_history.append(new_state)
            
        except Exception as e:
            if self.config.debug_mode:
                print(f"Error in attention analysis: {e}")
            self.attention_state = "Unknown"
    
    def _analyze_posture(self, pose_landmarks, frame: np.ndarray):
        """Analyze posture using pose landmarks."""
        try:
            # Extract key landmarks
            nose = pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
            
            # Check visibility
            min_visibility = 0.5
            if (left_shoulder.visibility < min_visibility or 
                right_shoulder.visibility < min_visibility or
                left_hip.visibility < min_visibility or
                right_hip.visibility < min_visibility):
                self.posture_state = "Partially Visible"
                return
            
            # Calculate posture metrics
            shoulder_angle = abs(math.degrees(math.atan2(
                right_shoulder.y - left_shoulder.y,
                right_shoulder.x - left_shoulder.x
            )))
            
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
            hip_center_x = (left_hip.x + right_hip.x) / 2
            hip_center_y = (left_hip.y + right_hip.y) / 2
            
            torso_angle = abs(math.degrees(math.atan2(
                shoulder_center_y - hip_center_y,
                shoulder_center_x - hip_center_x
            )))
            
            # Score posture
            shoulder_score = min(10, shoulder_angle / 3)
            torso_score = min(10, torso_angle / 5)
            total_score = shoulder_score * 0.5 + torso_score * 0.5
            
            # Classify posture
            if total_score < 3:
                self.posture_state = "Excellent"
            elif total_score < 5:
                self.posture_state = "Good"
            elif total_score < 7:
                self.posture_state = "Fair"
            else:
                self.posture_state = "Poor"
            
            self.posture_history.append(self.posture_state)
            
        except Exception as e:
            if self.config.debug_mode:
                print(f"Error in posture analysis: {e}")
            self.posture_state = "Unknown"
    
    def _analyze_movement(self, pose_landmarks, frame: np.ndarray):
        """Analyze movement patterns."""
        try:
            # Get key points
            nose = pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            left_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            # Create position vectors
            current_pos = np.array([
                nose.x, nose.y, nose.z,
                left_wrist.x, left_wrist.y, left_wrist.z,
                right_wrist.x, right_wrist.y, right_wrist.z,
                (left_shoulder.x + right_shoulder.x) / 2,
                (left_shoulder.y + right_shoulder.y) / 2,
                (left_shoulder.z + right_shoulder.z) / 2
            ])
            
            # Calculate movement
            if len(self.movement_history) > 0:
                prev_pos = self.movement_history[-1]
                pos_diff = current_pos - prev_pos
                movement_magnitude = np.linalg.norm(pos_diff)
                self.movement_magnitude.append(movement_magnitude)
                
                # Calculate average movement
                recent_movements = list(self.movement_magnitude)[-10:]
                avg_movement = sum(recent_movements) / len(recent_movements)
                
                # Classify movement level
                if avg_movement < 0.01:
                    self.movement_level = "Still"
                elif avg_movement < 0.03:
                    self.movement_level = "Low"
                elif avg_movement < 0.07:
                    self.movement_level = "Moderate"
                else:
                    self.movement_level = "High"
            
            self.movement_history.append(current_pos)
            
        except Exception as e:
            if self.config.debug_mode:
                print(f"Error in movement analysis: {e}")
            self.movement_level = "Unknown"
    
    def _analyze_fatigue(self, ear: float, threshold: float):
        """Analyze fatigue based on blink patterns and eye openness."""
        # Calculate current blink rate
        time_since_reset = time.time() - self.last_reset_time
        if time_since_reset > 0:
            current_blink_rate = self.blink_count / time_since_reset * 60.0
        else:
            current_blink_rate = 0
        
        # Check for extended eye closure
        extended_eye_closure = self.eye_closed_frames > self.config.eye_ar_consec_frames_closed * 5
        
        # Check for abnormal blink rate
        abnormal_blink_rate = False
        if len(self.blink_duration) > 5:
            avg_duration = sum(self.blink_duration) / len(self.blink_duration)
            recent_duration = sum(list(self.blink_duration)[-3:]) / 3
            if recent_duration > avg_duration * 1.3:
                abnormal_blink_rate = True
        
        # Determine fatigue level
        fatigue_indicators = sum([
            1 if extended_eye_closure else 0,
            1 if abnormal_blink_rate else 0,
            1 if self.drowsiness_frames > 10 else 0
        ])
        
        if fatigue_indicators >= 3:
            self.fatigue_level = "Severe"
        elif fatigue_indicators >= 2:
            self.fatigue_level = "Moderate"
        elif fatigue_indicators >= 1:
            self.fatigue_level = "Mild"
        else:
            self.fatigue_level = "Normal"
    
    def _collect_video_data(self, ear: float, threshold: float):
        """Collect video analysis data for output."""
        current_time = time.time()
        
        # Calculate blink rate
        time_since_reset = max(1, (current_time - self.last_reset_time))
        blink_rate = self.blink_count / time_since_reset * 60.0
        
        # Calculate additional metrics
        avg_blink_duration = sum(self.blink_duration) / max(1, len(self.blink_duration)) if self.blink_duration else 0
        avg_blink_interval = sum(self.blink_intervals) / max(1, len(self.blink_intervals)) if self.blink_intervals else 0
        
        # Calculate eye asymmetry
        eye_asymmetry = 0.0
        if len(self.left_ear_values) > 5 and len(self.right_ear_values) > 5:
            left_avg = sum(self.left_ear_values) / len(self.left_ear_values)
            right_avg = sum(self.right_ear_values) / len(self.right_ear_values)
            eye_asymmetry = abs(left_avg - right_avg) / max((left_avg + right_avg) / 2, 0.001)
        
        # Create data point
        data_point = {
            'timestamp': current_time,
            'emotion': self.current_emotion,
            'emotion_scores': self.emotion_scores.copy(),
            'attention_state': self.attention_state,
            'posture_state': self.posture_state,
            'movement_level': self.movement_level,
            'blink_rate': blink_rate,
            'total_blinks': self.total_blink_count,
            'ear': ear,
            'ear_threshold': threshold,
            'eye_asymmetry': eye_asymmetry,
            'blink_duration': avg_blink_duration,
            'blink_interval': avg_blink_interval,
            'fatigue_level': self.fatigue_level,
            'drowsiness_score': self.drowsiness_frames / max(1, len(self.ear_values)),
            'fps': self.performance_tracker.current_fps,
            'object_detections': getattr(self, 'current_detections', []),
            'person_tracking': self._get_person_tracking_data()
        }
        
        self.data_collector.add_video_data(data_point)
    
    def _is_face_in_main_person_region(self, face_landmarks, main_person: dict) -> bool:
        """
        Check if face landmarks are within the main person's bounding box.
        
        Args:
            face_landmarks: MediaPipe face landmarks
            main_person: Main person tracking data
            
        Returns:
            True if face is within main person's region
        """
        if not main_person or 'bbox' not in main_person:
            return True  # If no main person, process all faces
        
        # Get main person bounding box
        x1, y1, x2, y2 = main_person['bbox']
        
        # Get face center from landmarks
        face_center_x = 0
        face_center_y = 0
        landmark_count = 0
        
        for landmark in face_landmarks.landmark:
            face_center_x += landmark.x
            face_center_y += landmark.y
            landmark_count += 1
        
        if landmark_count == 0:
            return True
        
        face_center_x /= landmark_count
        face_center_y /= landmark_count
        
        # Convert to pixel coordinates (assuming 640x480 frame)
        frame_width, frame_height = 640, 480
        face_pixel_x = face_center_x * frame_width
        face_pixel_y = face_center_y * frame_height
        
        # Check if face center is within main person's bounding box
        return (x1 <= face_pixel_x <= x2 and y1 <= face_pixel_y <= y2)
    
    def _get_person_tracking_data(self) -> dict:
        """Get person tracking information."""
        if not hasattr(self, 'object_detector') or not self.object_detector:
            return {
                'total_persons': 0,
                'main_person_id': None,
                'main_person_confidence': 0.0,
                'has_main_person': False
            }
        
        tracking_info = self.object_detector.get_person_tracking_info()
        main_person = self.object_detector.get_main_person()
        
        return {
            'total_persons': tracking_info['total_persons'],
            'main_person_id': tracking_info['main_person_id'],
            'main_person_confidence': tracking_info['main_person_confidence'],
            'has_main_person': main_person is not None,
            'main_person_bbox': main_person['bbox'] if main_person else None,
            'person_detections': tracking_info['person_detections']
        }
    
    def get_blink_statistics(self) -> dict:
        """Get comprehensive blink statistics."""
        current_time = time.time()
        time_since_reset = max(1, (current_time - self.last_reset_time))
        
        # Calculate current blink rate
        current_blink_rate = self.blink_count / time_since_reset * 60.0
        
        # Calculate average metrics
        avg_blink_duration = sum(self.blink_duration) / max(1, len(self.blink_duration)) if self.blink_duration else 0
        avg_blink_interval = sum(self.blink_intervals) / max(1, len(self.blink_intervals)) if self.blink_intervals else 0
        
        # Calculate eye asymmetry
        eye_asymmetry = 0.0
        if len(self.left_ear_values) > 5 and len(self.right_ear_values) > 5:
            left_avg = sum(self.left_ear_values) / len(self.left_ear_values)
            right_avg = sum(self.right_ear_values) / len(self.right_ear_values)
            eye_asymmetry = abs(left_avg - right_avg) / max((left_avg + right_avg) / 2, 0.001)
        
        # Calculate blink rate statistics
        if len(self.blink_intervals) > 1:
            intervals = list(self.blink_intervals)
            min_interval = min(intervals)
            max_interval = max(intervals)
            std_interval = np.std(intervals) if len(intervals) > 1 else 0
        else:
            min_interval = max_interval = std_interval = 0
        
        return {
            'total_blinks': self.total_blink_count,
            'current_blink_rate': current_blink_rate,
            'average_blink_duration': avg_blink_duration,
            'average_blink_interval': avg_blink_interval,
            'min_blink_interval': min_interval,
            'max_blink_interval': max_interval,
            'blink_interval_std': std_interval,
            'eye_asymmetry': eye_asymmetry,
            'drowsiness_alerts': self.drowsiness_alerts,
            'fatigue_level': self.fatigue_level,
            'calibration_status': self.is_calibrated,
            'current_ear': sum(self.ear_values) / max(1, len(self.ear_values)) if self.ear_values else 0,
            'adaptive_threshold': self.adaptive_threshold,
            'eye_state': 'closed' if self.eye_closed else 'open',
            'session_duration': time_since_reset
        }
    
    def _display_results(self, frame: np.ndarray, ear: float, threshold: float):
        """Display analysis results on frame."""
        h, w, _ = frame.shape
        
        # Create overlay
        overlay = frame.copy()
        
        # Main panel
        cv2.rectangle(overlay, (10, 10), (340, 320), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (10, 10), (340, 320), (100, 100, 100), 1)
        
        # Title
        cv2.rectangle(frame, (10, 10), (340, 40), (40, 40, 40), -1)
        cv2.putText(frame, "Video Analysis", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display results
        y_pos = 70
        
        # Emotion
        emotion_color = ColorUtils.get_emotion_color(self.current_emotion)
        cv2.putText(frame, f"Emotion: {self.current_emotion}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, emotion_color, 2)
        
        # Attention
        y_pos += 30
        attention_color = ColorUtils.get_attention_color(self.attention_state)
        cv2.putText(frame, f"Attention: {self.attention_state}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, attention_color, 2)
        
        # Posture
        y_pos += 30
        posture_color = ColorUtils.get_posture_color(self.posture_state)
        cv2.putText(frame, f"Posture: {self.posture_state}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, posture_color, 2)
        
        # Movement
        y_pos += 30
        cv2.putText(frame, f"Movement: {self.movement_level}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Enhanced Blink Analysis Section
        y_pos += 30
        time_since_reset = max(1, (time.time() - self.last_reset_time))
        blink_rate = self.blink_count / time_since_reset * 60.0
        
        # Blink rate with color coding
        if blink_rate < 10:
            blink_color = (0, 0, 255)  # Red - too low
        elif blink_rate > 30:
            blink_color = (0, 165, 255)  # Orange - too high
        else:
            blink_color = (0, 255, 0)  # Green - normal
        
        cv2.putText(frame, f"Blink Rate: {blink_rate:.1f} bpm", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, blink_color, 2)
        
        # Total blinks
        y_pos += 25
        cv2.putText(frame, f"Total Blinks: {self.total_blink_count}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Blink duration and interval
        if self.blink_duration:
            avg_duration = sum(self.blink_duration) / len(self.blink_duration)
            cv2.putText(frame, f"Avg Duration: {avg_duration:.3f}s", (20, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        if self.blink_intervals:
            avg_interval = sum(self.blink_intervals) / len(self.blink_intervals)
            cv2.putText(frame, f"Avg Interval: {avg_interval:.1f}s", (20, y_pos + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Fatigue
        y_pos += 60
        fatigue_color = ColorUtils.get_emotion_color(self.fatigue_level)
        cv2.putText(frame, f"Fatigue: {self.fatigue_level}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fatigue_color, 2)
        
        # Person Tracking Info
        if hasattr(self, 'object_detector') and self.object_detector:
            tracking_info = self.object_detector.get_person_tracking_info()
            main_person = self.object_detector.get_main_person()
            
            y_pos += 30
            # Total person count
            cv2.putText(frame, f"People: {tracking_info['total_persons']}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Main person info
            if main_person:
                y_pos += 25
                cv2.putText(frame, f"Main Person: {main_person['id'][:8]}...", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                y_pos += 20
                cv2.putText(frame, f"Confidence: {tracking_info['main_person_confidence']:.2f}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                y_pos += 15
                cv2.putText(frame, f"Size: {main_person['size_ratio']:.3f}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            else:
                y_pos += 25
                cv2.putText(frame, "No main person detected", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Object Detection Info (non-person objects)
        if hasattr(self, 'current_detections') and self.current_detections:
            non_person_objects = [det for det in self.current_detections if det['class_name'] != 'person']
            if non_person_objects:
                y_pos += 30
                cv2.putText(frame, f"Other Objects: {len(non_person_objects)}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Status panel
        status_x = w - 220
        status_y = 10
        cv2.rectangle(frame, (status_x, status_y), (w-10, status_y+90), (0, 0, 0), -1)
        cv2.rectangle(frame, (status_x, status_y), (w-10, status_y+90), (100, 100, 100), 1)
        
        # FPS
        cv2.putText(frame, f"FPS: {self.performance_tracker.current_fps:.1f}", 
                   (status_x + 10, status_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Calibration status
        status = "Calibrated" if self.is_calibrated else f"Calibrating {min(100, len(self.ear_baseline)*2)}%"
        status_color = (0, 255, 0) if self.is_calibrated else (0, 165, 255)
        cv2.putText(frame, f"Status: {status}", 
                   (status_x + 10, status_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # Session time
        session_time = time.time() - self.session_start_time
        minutes, seconds = divmod(int(session_time), 60)
        hours, minutes = divmod(minutes, 60)
        time_str = f"{hours}h {minutes}m {seconds}s" if hours > 0 else f"{minutes}m {seconds}s"
        cv2.putText(frame, f"Session: {time_str}", 
                   (status_x + 10, status_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Enhanced Eye State Indicator
        self._draw_eye_state_indicator(frame, ear, threshold)
        
        # Blink Detection Graph
        if self.config.debug_mode:
            self._draw_blink_graph(frame, ear, threshold)
    
    def _draw_eye_state_indicator(self, frame: np.ndarray, ear: float, threshold: float):
        """Draw enhanced eye state indicator with blink detection."""
        h, w, _ = frame.shape
        
        # Eye state indicator position
        eye_state_x = w - 100
        eye_state_y = 130
        
        # Create eye state indicator background
        cv2.rectangle(frame, (eye_state_x - 70, eye_state_y - 40), 
                     (eye_state_x + 70, eye_state_y + 40), (40, 40, 40), -1)
        cv2.rectangle(frame, (eye_state_x - 70, eye_state_y - 40), 
                     (eye_state_x + 70, eye_state_y + 40), (100, 100, 100), 2)
        
        # Title
        cv2.putText(frame, "EYE STATE", (eye_state_x - 35, eye_state_y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw eye based on state
        if self.eye_closed:
            # Draw closed eye
            cv2.line(frame, (eye_state_x - 30, eye_state_y), 
                    (eye_state_x + 30, eye_state_y), (0, 0, 255), 4)
            state_text = "CLOSED"
            state_color = (0, 0, 255)
            
            # Add blink animation effect
            if hasattr(self, 'blink_start_time'):
                blink_duration = time.time() - self.blink_start_time
                if blink_duration < 0.2:  # First 200ms of blink
                    # Draw pulsing effect
                    pulse_intensity = int(255 * (1 - blink_duration / 0.2))
                    cv2.line(frame, (eye_state_x - 30, eye_state_y), 
                            (eye_state_x + 30, eye_state_y), (0, 0, pulse_intensity), 6)
        else:
            # Draw open eye
            cv2.ellipse(frame, (eye_state_x, eye_state_y), (30, 15), 
                       0, 0, 360, (0, 255, 0), 3)
            cv2.circle(frame, (eye_state_x, eye_state_y), 10, (0, 255, 0), -1)
            cv2.circle(frame, (eye_state_x, eye_state_y), 3, (0, 0, 0), -1)  # Pupil
            state_text = "OPEN"
            state_color = (0, 255, 0)
        
        # State text
        cv2.putText(frame, state_text, (eye_state_x - 25, eye_state_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 2)
        
        # EAR value and threshold
        cv2.putText(frame, f"EAR: {ear:.3f}", (eye_state_x - 25, eye_state_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        cv2.putText(frame, f"Thresh: {threshold:.3f}", (eye_state_x - 30, eye_state_y + 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        
        # Blink counter
        cv2.putText(frame, f"Blinks: {self.total_blink_count}", (eye_state_x - 30, eye_state_y + 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    def _draw_blink_graph(self, frame: np.ndarray, ear: float, threshold: float):
        """Draw real-time blink detection graph."""
        h, w, _ = frame.shape
        
        # Graph dimensions and position
        graph_width = 200
        graph_height = 100
        graph_x = w - graph_width - 10
        graph_y = h - graph_height - 10
        
        # Draw graph background
        cv2.rectangle(frame, (graph_x, graph_y), 
                     (graph_x + graph_width, graph_y + graph_height), (20, 20, 20), -1)
        cv2.rectangle(frame, (graph_x, graph_y), 
                     (graph_x + graph_width, graph_y + graph_height), (80, 80, 80), 1)
        
        # Title
        cv2.putText(frame, "EAR History", (graph_x + 5, graph_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Calculate y-scale
        if len(self.ear_values) > 0:
            y_min = max(0.001, min(self.min_ear, ear) * 0.8)
            y_max = max(self.max_ear, ear) * 1.2
        else:
            y_min = 0.0
            y_max = 0.5
        
        y_range = y_max - y_min
        
        # Draw threshold line
        if y_range > 0:
            threshold_y = graph_y + graph_height - int((threshold - y_min) / y_range * graph_height)
            threshold_y = max(graph_y, min(threshold_y, graph_y + graph_height))
            cv2.line(frame, (graph_x, threshold_y), 
                    (graph_x + graph_width, threshold_y), (0, 180, 255), 2)
        
        # Draw EAR values
        if len(self.ear_values) > 1:
            values_to_draw = list(self.ear_values)[-graph_width:] if len(self.ear_values) > graph_width else list(self.ear_values)
            
            if len(values_to_draw) > 1:
                # Draw line segments
                for i in range(1, len(values_to_draw)):
                    x1 = graph_x + int((i - 1) * graph_width / max(1, len(values_to_draw) - 1))
                    x2 = graph_x + int(i * graph_width / max(1, len(values_to_draw) - 1))
                    
                    y1 = graph_y + graph_height - int((values_to_draw[i-1] - y_min) / max(0.001, y_range) * graph_height)
                    y2 = graph_y + graph_height - int((values_to_draw[i] - y_min) / max(0.001, y_range) * graph_height)
                    
                    # Keep y within bounds
                    y1 = max(graph_y, min(y1, graph_y + graph_height))
                    y2 = max(graph_y, min(y2, graph_y + graph_height))
                    
                    # Color based on threshold
                    if values_to_draw[i] < threshold:
                        line_color = (0, 0, 255)  # Red for below threshold
                    else:
                        line_color = (0, 255, 0)  # Green for above threshold
                    
                    cv2.line(frame, (x1, y1), (x2, y2), line_color, 2)
        
        # Mark current value
        if y_range > 0:
            current_y = graph_y + graph_height - int((ear - y_min) / y_range * graph_height)
            current_y = max(graph_y, min(current_y, graph_y + graph_height))
            current_x = graph_x + graph_width - 5
            
            # Color based on current state
            if ear < threshold:
                point_color = (0, 0, 255)  # Red
            else:
                point_color = (0, 255, 0)  # Green
            
            cv2.circle(frame, (current_x, current_y), 4, point_color, -1)
            cv2.circle(frame, (current_x, current_y), 4, (255, 255, 255), 1)
    
    def cleanup(self):
        """Clean up resources."""
        # Stop emotion thread
        if hasattr(self, 'emotion_thread_active'):
            self.emotion_thread_active = False
            if hasattr(self, 'emotion_thread') and self.emotion_thread.is_alive():
                self.emotion_thread.join(timeout=1.0)
        
        # Close MediaPipe components
        self.face_mesh.close()
        self.pose.close()
        
        # Cleanup object detector
        if hasattr(self, 'object_detector') and self.object_detector:
            self.object_detector.cleanup()
        
        print("Video analyzer cleaned up")
