import time
import math
import cv2
import numpy as np
from scipy.spatial import distance as dist

try:
    import mediapipe as mp
    TASKS_AVAILABLE = True
except Exception:
    import mediapipe as mp
    TASKS_AVAILABLE = False

# --- Constants ---
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_INDICES = [61, 291, 0, 17]

DUMMY_VALUE = -999.0

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
    """
    Calculates Head Pose (Pitch, Yaw, Roll) + Visuals
    """
    try:
        # Landmarks
        nose = landmarks[1]
        left_eye_outer = landmarks[33]
        right_eye_outer = landmarks[263]
        mouth_center = landmarks[13] # Upper lip center roughly

        # Coords
        nx, ny = nose.x * w, nose.y * h
        lx, ly = left_eye_outer.x * w, left_eye_outer.y * h
        rx, ry = right_eye_outer.x * w, right_eye_outer.y * h
        mx, my = mouth_center.x * w, mouth_center.y * h

        # --- VISUAL: Draw Roll Line (Eye to Eye) ---
        # This helps you see what "Roll" is measuring
        cv2.line(overlay, (int(lx), int(ly)), (int(rx), int(ry)), (255, 0, 255), 2)

        # 1. ROLL (Head Tilt)
        # Calculate angle of the eye-line relative to horizontal
        dY = ry - ly
        dX = rx - lx
        # Result is usually 0 if eyes are level
        roll = math.degrees(math.atan2(dY, dX))

        # 2. YAW (Turn)
        dist_l = math.hypot(nx - lx, ny - ly)
        dist_r = math.hypot(nx - rx, ny - ry)
        yaw = ((dist_l - dist_r) / (dist_l + dist_r + 1e-6)) * 150

        # 3. PITCH (Nod)
        ex, ey = (lx + rx) / 2, (ly + ry) / 2
        dist_nose_eyes = math.hypot(nx - ex, ny - ey)
        dist_nose_mouth = math.hypot(nx - mx, ny - my)
        pitch = (dist_nose_eyes / (dist_nose_mouth + 1e-6) - 1.0) * 100

        return pitch, yaw, roll

    except Exception:
        return DUMMY_VALUE, DUMMY_VALUE, DUMMY_VALUE

def calculate_slump_geometry(pose_landmarks, face_landmarks, w, h, overlay):
    """
    Calculates Slump and Shoulder Tilt (R_Tilt)
    """
    if not pose_landmarks: return DUMMY_VALUE, DUMMY_VALUE

    try:
        p_nose = pose_landmarks[0]
        # MP Landmark 11 = Person's Left Shoulder (Viewer's Right)
        # MP Landmark 12 = Person's Right Shoulder (Viewer's Left)
        p_left_sh = pose_landmarks[11] 
        p_right_sh = pose_landmarks[12]

        x_n, y_n = int(p_nose.x * w), int(p_nose.y * h)
        x_l, y_l = int(p_left_sh.x * w), int(p_left_sh.y * h)
        x_r, y_r = int(p_right_sh.x * w), int(p_right_sh.y * h)
        mx, my = int((x_l + x_r) / 2), int((y_l + y_r) / 2)

        # Draw Lines
        cv2.line(overlay, (x_l, y_l), (x_r, y_r), (255, 0, 0), 3) # Blue Shoulder Line
        cv2.line(overlay, (x_n, y_n), (mx, my), (0, 255, 255), 3) # Yellow Vert Line

        # --- FIX FOR R_TILT ---
        # We want the angle of the line connecting shoulders relative to horizontal.
        # To get 0 degrees when straight, we must calculate Vector = Left(Screen) - Right(Screen)
        # Landmark 12 (Person Right) is usually on Left of Screen (Low X)
        # Landmark 11 (Person Left) is usually on Right of Screen (High X)
        
        # Vector pointing from Left-Screen-Side (12) to Right-Screen-Side (11)
        dY = y_l - y_r 
        dX = x_l - x_r 
        
        # Now atan2(0, PositiveNumber) = 0 degrees.
        r_tilt = math.degrees(math.atan2(dY, dX))

        # --- SLUMP ---
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

def run_demo(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    
    mp_face_mesh = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles 
    
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

            ear = mar = pitch = yaw = roll = d_slump = r_tilt = DUMMY_VALUE
            
            if face_disp:
                flms = face_results.multi_face_landmarks[0].landmark
                pts = [(lm.x * w, lm.y * h) for lm in flms]
                ear = (eye_aspect_ratio([pts[i] for i in LEFT_EYE_INDICES]) + 
                       eye_aspect_ratio([pts[i] for i in RIGHT_EYE_INDICES])) / 2.0
                mar = mouth_aspect_ratio([pts[i] for i in MOUTH_INDICES])
                
                # --- CALCULATE HEAD POSE ---
                pitch, yaw, roll = calculate_geometric_head_pose(flms, w, h, overlay)

            if pose_disp:
                plms = pose_results.pose_landmarks.landmark
                flms = face_results.multi_face_landmarks[0].landmark if face_disp else None
                # --- CALCULATE BODY POSE ---
                d_slump, r_tilt = calculate_slump_geometry(plms, flms, w, h, overlay)

            # Display
            lines = [
                f"EAR: {ear:.2f}" if ear != DUMMY_VALUE else "EAR: N/A",
                f"MAR: {mar:.2f}" if mar != DUMMY_VALUE else "MAR: N/A",
                f"Pitch: {pitch:.1f}" if pitch != DUMMY_VALUE else "Pitch: N/A",
                f"Yaw: {yaw:.1f}" if yaw != DUMMY_VALUE else "Yaw: N/A",
                f"Roll (Head): {roll:.1f}" if roll != DUMMY_VALUE else "Roll: N/A",
                f"Slump: {d_slump:.2f}" if d_slump != DUMMY_VALUE else "Slump: N/A",
                f"R_Tilt (Body): {r_tilt:.1f}" if r_tilt != DUMMY_VALUE else "R_Tilt: N/A",
                f"Face: {face_disp} | Body: {pose_disp}"
            ]
            
            for i, line in enumerate(lines):
                cv2.putText(overlay, line, (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Final Stable System", overlay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_demo()