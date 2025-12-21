import time
import math
import cv2
import numpy as np
import os
from scipy.spatial import distance as dist

try:
    import mediapipe as mp
    TASKS_AVAILABLE = True
except Exception:
    import mediapipe as mp
    TASKS_AVAILABLE = False

# --- Config & Constants ---
OUTPUT_ROOT = "Datasets_ReRecorded"
CLASSES = {
    0: "Awake",
    1: "Sleep",
    2: "Yawning"
}

# MediaPipe Indices
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_INDICES = [61, 291, 0, 17]

# Set to 0.0 to match Scaler expectations
DUMMY_VALUE = 0.0

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

        # Visual Line for eyes
        cv2.line(overlay, (int(lx), int(ly)), (int(rx), int(ry)), (255, 0, 255), 2)

        # Roll
        dY = ry - ly
        dX = rx - lx
        roll = math.degrees(math.atan2(dY, dX))

        # Yaw
        dist_l = math.hypot(nx - lx, ny - ly)
        dist_r = math.hypot(nx - rx, ny - ry)
        yaw = ((dist_l - dist_r) / (dist_l + dist_r + 1e-6)) * 150

        # Pitch
        ex, ey = (lx + rx) / 2, (ly + ry) / 2
        dist_nose_eyes = math.hypot(nx - ex, ny - ey)
        dist_nose_mouth = math.hypot(nx - mx, ny - my)
        pitch = (dist_nose_eyes / (dist_nose_mouth + 1e-6) - 1.0) * 100

        return pitch, yaw, roll
    except Exception:
        return DUMMY_VALUE, DUMMY_VALUE, DUMMY_VALUE

def calculate_slump_geometry(pose_landmarks, face_landmarks, w, h, overlay):
    if not pose_landmarks: return DUMMY_VALUE, DUMMY_VALUE
    try:
        p_nose = pose_landmarks[0]
        p_left_sh = pose_landmarks[11] 
        p_right_sh = pose_landmarks[12]

        x_n, y_n = int(p_nose.x * w), int(p_nose.y * h)
        x_l, y_l = int(p_left_sh.x * w), int(p_left_sh.y * h)
        x_r, y_r = int(p_right_sh.x * w), int(p_right_sh.y * h)
        mx, my = int((x_l + x_r) / 2), int((y_l + y_r) / 2)

        # Draw shoulders and spine
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
            d_slump = DUMMY_VALUE

        return d_slump, r_tilt
    except Exception:
        return DUMMY_VALUE, DUMMY_VALUE

# --- Main Data Collection Function ---
def run_data_collection(camera_index=0):
    # 1. Setup Directories
    for label_name in CLASSES.values():
        path = os.path.join(OUTPUT_ROOT, label_name)
        os.makedirs(path, exist_ok=True)
        print(f"Ensured folder exists: {path}")

    cap = cv2.VideoCapture(camera_index)
    
    mp_face_mesh = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles 

    # State Variables
    current_label_id = 0 # Default Awake
    frames_to_record = 0 # Counter for burst mode
    total_captured = {0: 0, 1: 0, 2: 0} # Count per session

    print("\n--- INSTRUCTIONS ---")
    print("Press '0' -> Switch to AWAKE Mode")
    print("Press '1' -> Switch to SLEEP Mode")
    print("Press '2' -> Switch to YAWNING Mode")
    print("Press 'SPACE' -> Capture 20 Frames (Burst)")
    print("Press 'q' -> Quit")
    print("--------------------\n")
    
    with mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1) as face_mesh, \
         mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            overlay = frame.copy()

            # Process MediaPipe
            face_results = face_mesh.process(rgb)
            pose_results = pose.process(rgb)

            face_disp = 1 if face_results.multi_face_landmarks else 0
            pose_disp = 1 if pose_results.pose_landmarks else 0
            
            # Draw Face Mesh
            if face_disp:
                for face_landmarks in face_results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=overlay,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

            # Calculate Features
            ear = mar = pitch = yaw = roll = d_slump = r_tilt = DUMMY_VALUE
            if face_disp:
                flms = face_results.multi_face_landmarks[0].landmark
                pts = [(lm.x * w, lm.y * h) for lm in flms]
                ear = (eye_aspect_ratio([pts[i] for i in LEFT_EYE_INDICES]) + 
                       eye_aspect_ratio([pts[i] for i in RIGHT_EYE_INDICES])) / 2.0
                mar = mouth_aspect_ratio([pts[i] for i in MOUTH_INDICES])
                pitch, yaw, roll = calculate_geometric_head_pose(flms, w, h, overlay)

            if pose_disp:
                plms = pose_results.pose_landmarks.landmark
                flms = face_results.multi_face_landmarks[0].landmark if face_disp else None
                d_slump, r_tilt = calculate_slump_geometry(plms, flms, w, h, overlay)

            # --- RECORDING LOGIC ---
            if frames_to_record > 0:
                # Save Raw Frame (Not overlay)
                label_name = CLASSES[current_label_id]
                timestamp = int(time.time() * 1000)
                filename = f"{OUTPUT_ROOT}/{label_name}/{timestamp}_{frames_to_record}.jpg"
                cv2.imwrite(filename, frame)
                
                frames_to_record -= 1
                total_captured[current_label_id] += 1
                cv2.putText(overlay, f"RECORDING... {frames_to_record}", (w//2 - 100, h - 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # --- DISPLAY ALL FEATURE VALUES (FINAL_COLUMNS) ---
            # Text generation logic
            if face_disp:
                txt_ear = f"EAR:   {ear:.2f}"
                txt_mar = f"MAR:   {mar:.2f}"
                txt_pitch = f"PITCH: {pitch:.1f} (Keep ~0)"
                txt_yaw   = f"YAW:   {yaw:.1f}"
                txt_roll  = f"ROLL:  {roll:.1f}"
            else:
                txt_ear = txt_mar = txt_pitch = txt_yaw = txt_roll = "N/A (No Face)"

            if pose_disp:
                txt_slump = f"SLUMP: {d_slump:.2f}"
                txt_tilt  = f"R_TILT:{r_tilt:.1f}"
            else:
                txt_slump = txt_tilt = "N/A (No Body)"

            # Prepare list of lines to draw
            info_lines = [
                f"MODE:  {CLASSES[current_label_id].upper()}",
                f"SAVED: {total_captured[current_label_id]}",
                "----------------",
                txt_ear,
                txt_mar,
                txt_pitch,
                txt_yaw,
                txt_roll,
                txt_slump,
                txt_tilt
            ]
            
            # Draw lines
            start_y = 30
            for i, line in enumerate(info_lines):
                # Color logic: Yellow for Mode/Saved, Green for valid features, Gray for N/A
                if "MODE" in line or "SAVED" in line:
                    color = (0, 255, 255) # Yellow
                elif "N/A" in line:
                    color = (150, 150, 150) # Gray
                else:
                    color = (0, 255, 0) # Green
                
                cv2.putText(overlay, line, (10, start_y + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Bottom Instructions
            cv2.putText(overlay, "[0]Awake [1]Sleep [2]Yawn [SPACE]Record [Q]Quit", (10, h-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Data Collection Tool", overlay)

            # --- KEY CONTROLS ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('0'):
                current_label_id = 0
            elif key == ord('1'):
                current_label_id = 1
            elif key == ord('2'):
                current_label_id = 2
            elif key == ord(' '):
                frames_to_record = 20 # Trigger burst

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_data_collection()