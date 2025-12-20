import time
import math
import cv2
import numpy as np
import os
import joblib
import tensorflow as tf
from scipy.spatial import distance as dist
import glob # <--- NEW IMPORT FOR MODEL LOADING
from collections import deque  # <--- NEW IMPORT FOR SMOOTHING

try:
    import mediapipe as mp
    TASKS_AVAILABLE = True
except Exception:
    print("Warning: MediaPipe not installed or conflicting file 'mediapipe.py' found.")
    TASKS_AVAILABLE = False

# --- Constants & Config ---
MODEL_FOLDER = './model'
SCALER_PATH = 'scaler.pkl'

LABELS = {0: "AWAKE", 1: "SLEEP", 2: "YAWNING"}
COLORS = {0: (0, 255, 0), 1: (0, 0, 255), 2: (0, 165, 255)} # Green, Red, Orange

# MediaPipe Indices
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_INDICES = [61, 291, 0, 17]

DEFAULT_SLUMP = 0.8 
DEFAULT_TILT = 0.0
DUMMY_VALUE = 0.0

# --- SENSITIVITY CONFIG ---
# Buffer Length: 30 frames is approx 1 second of "Memory".
# Increase to 60 for more stability (slower response).
# Decrease to 15 for faster response (less stable).
SMOOTHING_BUFFER_SIZE = 15 

# --- Load Model & Scaler ---
print(f"Loading Scaler from {SCALER_PATH}...")
try:
    scaler = joblib.load(SCALER_PATH)
    print("Scaler Loaded!")
except Exception as e:
    print(f"FATAL ERROR: Could not load scaler.pkl. {e}")
    exit()

print(f"Searching for model in {MODEL_FOLDER}...")

# 1. Find all .keras files in the folder
model_files = glob.glob(os.path.join(MODEL_FOLDER, "*.keras"))

if not model_files:
    print(f"FATAL ERROR: No .keras model found in {MODEL_FOLDER}")
    exit()

# 2. Pick the first one found (or you could sort by date/name)
MODEL_PATH = model_files[0]
print(f"Found Model: {MODEL_PATH}")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model Loaded Successfully!")
except Exception as e:
    print(f"FATAL ERROR: Could not load model. {e}")
    exit()


# --- Helper Functions ---
def eye_aspect_ratio(eye_coords):
    p1, p2, p3, p4, p5, p6 = eye_coords
    vertical_1 = dist.euclidean(p2, p6)
    vertical_2 = dist.euclidean(p3, p5)
    horizontal = dist.euclidean(p1, p4)
    if horizontal == 0: return 0.001
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

def mouth_aspect_ratio(mouth_coords):
    p1_h, p4_h, p2_v, p6_v = mouth_coords
    vertical = dist.euclidean(p2_v, p6_v)
    horizontal = dist.euclidean(p1_h, p4_h)
    if horizontal == 0: return 0.001
    return vertical / horizontal

def calculate_geometric_head_pose(landmarks, w, h, overlay):
    try:
        nose = landmarks[1]
        left_eye_outer = landmarks[33]
        right_eye_outer = landmarks[263]
        mouth_center = landmarks[13]

        nx, ny = nose.x * w, nose.y * h
        lx, ly = left_eye_outer.x * w, left_eye_outer.y * h
        rx, ry = right_eye_outer.x * w, right_eye_outer.y * h
        mx, my = mouth_center.x * w, mouth_center.y * h

        cv2.line(overlay, (int(lx), int(ly)), (int(rx), int(ry)), (255, 0, 255), 2)

        dY = ry - ly
        dX = rx - lx
        roll = math.degrees(math.atan2(dY, dX))

        dist_l = math.hypot(nx - lx, ny - ly)
        dist_r = math.hypot(nx - rx, ny - ry)
        yaw = ((dist_l - dist_r) / (dist_l + dist_r + 1e-6)) * 150

        ex, ey = (lx + rx) / 2, (ly + ry) / 2
        dist_nose_eyes = math.hypot(nx - ex, ny - ey)
        dist_nose_mouth = math.hypot(nx - mx, ny - my)
        pitch = (dist_nose_eyes / (dist_nose_mouth + 1e-6) - 1.0) * 100

        return pitch, yaw, roll
    except Exception:
        return 0, 0, 0

def calculate_slump_geometry(pose_landmarks, face_landmarks, w, h, overlay):
    if not pose_landmarks: return DEFAULT_SLUMP, DEFAULT_TILT

    try:
        p_nose = pose_landmarks[0]
        p_left_sh = pose_landmarks[11] 
        p_right_sh = pose_landmarks[12]

        x_n, y_n = int(p_nose.x * w), int(p_nose.y * h)
        x_l, y_l = int(p_left_sh.x * w), int(p_left_sh.y * h)
        x_r, y_r = int(p_right_sh.x * w), int(p_right_sh.y * h)
        mx, my = int((x_l + x_r) / 2), int((y_l + y_r) / 2)

        cv2.line(overlay, (x_l, y_l), (x_r, y_r), (255, 0, 0), 3) 
        cv2.line(overlay, (x_n, y_n), (mx, my), (0, 255, 255), 3)

        dY = y_l - y_r 
        dX = x_l - x_r 
        r_tilt = math.degrees(math.atan2(dY, dX))

        if face_landmarks:
            chin_y = face_landmarks[152].y * h
            head_top_y = face_landmarks[10].y * h
            face_h = abs(chin_y - head_top_y)
            if face_h < 1: face_h = 1
            d_slump = (my - y_n) / face_h 
        else:
            d_slump = DEFAULT_SLUMP

        return d_slump, r_tilt

    except Exception:
        return DEFAULT_SLUMP, DEFAULT_TILT

def run_demo(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    
    mp_face_mesh = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles 
    
    # --- Persistence Variables ---
    current_status = "INITIALIZING"
    current_conf = 0.0
    current_color = (200, 200, 200)

    # --- SMOOTHING BUFFER ---
    # Stores the raw probabilities (e.g., [0.1, 0.8, 0.1]) of the last X frames
    prediction_buffer = deque(maxlen=SMOOTHING_BUFFER_SIZE)

    with mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1) as face_mesh, \
         mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            overlay = frame.copy()

            face_results = face_mesh.process(rgb)
            pose_results = pose.process(rgb)

            face_disp = 1 if face_results.multi_face_landmarks else 0
            pose_disp = 1 if pose_results.pose_landmarks else 0
            
            # --- Draw Face Mesh ---
            if face_disp:
                for face_landmarks in face_results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=overlay,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

            # Initialize Features
            ear = mar = pitch = yaw = roll = 0.0
            d_slump = DEFAULT_SLUMP
            r_tilt = DEFAULT_TILT
            
            # --- 1. CALCULATE FACE FEATURES ---
            if face_disp:
                flms = face_results.multi_face_landmarks[0].landmark
                pts = [(lm.x * w, lm.y * h) for lm in flms]
                ear = (eye_aspect_ratio([pts[i] for i in LEFT_EYE_INDICES]) + 
                       eye_aspect_ratio([pts[i] for i in RIGHT_EYE_INDICES])) / 2.0
                mar = mouth_aspect_ratio([pts[i] for i in MOUTH_INDICES])
                pitch, yaw, roll = calculate_geometric_head_pose(flms, w, h, overlay)

            # --- 2. CALCULATE BODY FEATURES ---
            if pose_disp:
                plms = pose_results.pose_landmarks.landmark
                flms = face_results.multi_face_landmarks[0].landmark if face_disp else None
                d_slump, r_tilt = calculate_slump_geometry(plms, flms, w, h, overlay)

            # --- 3. PREDICTION LOGIC (ROLLING AVERAGE) ---
            
            if not face_disp:
                # Immediate override if no face
                # We clear the buffer so old "awake" frames don't delay the "No Detection" alert
                prediction_buffer.clear()
                current_status = "NO DETECTION"
                current_color = (100, 100, 100)
                current_conf = 0.0
            else:
                # Prepare Vector
                input_features = np.array([[ear, mar, pitch, yaw, roll, d_slump, r_tilt]])
                scaled_features = scaler.transform(input_features)
                
                # Predict Probabilities (e.g., [0.1, 0.8, 0.1])
                preds = model.predict(scaled_features, verbose=0)[0] 
                
                # Add to Buffer
                prediction_buffer.append(preds)
                
                # Calculate Average across the buffer
                # Axis 0 means average down the columns (average all 'Awake' scores, etc.)
                avg_preds = np.mean(prediction_buffer, axis=0)
                
                # Get the label with the highest AVERAGE probability
                idx = np.argmax(avg_preds)
                
                current_status = LABELS[idx]
                current_conf = avg_preds[idx]
                current_color = COLORS[idx]

            # --- 4. DISPLAY ---
            # Big Result on Top
            cv2.putText(overlay, f"{current_status} ({current_conf*100:.1f}%)", (50, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, current_color, 4)

            # Detailed Stats List
            lines = [
                f"EAR:   {ear:.2f}" if face_disp else "EAR: N/A",
                f"MAR:   {mar:.2f}" if face_disp else "MAR: N/A",
                f"Pitch: {pitch:.1f}" if face_disp else "Pitch: N/A",
                f"Yaw:   {yaw:.1f}" if face_disp else "Yaw: N/A",
                f"Roll:  {roll:.1f}" if face_disp else "Roll: N/A",
                f"Slump: {d_slump:.2f}" if pose_disp else f"Slump: {d_slump} (Default)",
                f"Tilt:  {r_tilt:.1f}" if pose_disp else f"Tilt: {r_tilt} (Default)",
                f"Face: {face_disp} | Body: {pose_disp}",
                "----------------",
                f"STATUS: {current_status}"
            ]
            
            # Draw stats
            for i, line in enumerate(lines):
                color = (0, 0, 0) 
                if "STATUS" in line:
                    color = current_color
                
                y_pos = h - 280 + (i * 25)
                cv2.putText(overlay, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("Final Stable System", overlay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_demo()