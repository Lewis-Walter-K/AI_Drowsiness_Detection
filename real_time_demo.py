import os
import time
import math
import cv2
import numpy as np
from scipy.spatial import distance as dist

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from mediapipe import Image as MpImage
    TASKS_AVAILABLE = True
except Exception:
    import mediapipe as mp
    TASKS_AVAILABLE = False

# --- Constants / indices (same as notebook) ---
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_INDICES = [61, 291, 0, 17]
HEAD_POSE_INDICES = [1, 33, 263, 61, 291, 152]

MODEL_3D_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0),
    (0.0, -330.0, -65.0)
], dtype=np.float32)

DUMMY_VALUE = -999.0


def eye_aspect_ratio(eye_coords):
    p1, p2, p3, p4, p5, p6 = eye_coords
    vertical_1 = dist.euclidean(p2, p6)
    vertical_2 = dist.euclidean(p3, p5)
    horizontal = dist.euclidean(p1, p4)
    if horizontal == 0:
        return 0.001
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def mouth_aspect_ratio(mouth_coords):
    p1_h, p4_h, p2_v, p6_v = mouth_coords
    vertical = dist.euclidean(p2_v, p6_v)
    horizontal = dist.euclidean(p1_h, p4_h)
    if horizontal == 0:
        return 0.001
    return vertical / horizontal


def _compute_fallback_head_pose(landmarks_list, w, h):
    try:
        xs = [float(lm.x) * w for lm in landmarks_list]
        ys = [float(lm.y) * h for lm in landmarks_list]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        face_w = max_x - min_x if (max_x - min_x) > 1e-6 else float(w)
        face_h = max_y - min_y if (max_y - min_y) > 1e-6 else float(h)

        def safe_pt(idx):
            try:
                p = landmarks_list[idx]
                return float(p.x) * w, float(p.y) * h
            except Exception:
                return (min_x + face_w/2.0, min_y + face_h/2.0)

        nose_x, nose_y = safe_pt(1)
        left_eye_x, left_eye_y = safe_pt(33)
        right_eye_x, right_eye_y = safe_pt(263)

        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0

        yaw = ((nose_x - center_x) / (face_w + 1e-6)) * 60.0
        pitch = ((center_y - nose_y) / (face_h + 1e-6)) * 40.0
        roll_rad = math.atan2((right_eye_y - left_eye_y), (right_eye_x - left_eye_x + 1e-6))
        roll = math.degrees(roll_rad)

        yaw = max(min(yaw, 120.0), -120.0)
        pitch = max(min(pitch, 120.0), -120.0)
        roll = max(min(roll, 120.0), -120.0)

        return yaw, pitch, roll
    except Exception:
        return DUMMY_VALUE, DUMMY_VALUE, DUMMY_VALUE


def get_head_pose_from_face_mesh(landmarks, w, h):
    try:
        image_points = []
        for i in HEAD_POSE_INDICES:
            lm = landmarks[i]
            x = float(lm.x) * float(w)
            y = float(lm.y) * float(h)
            image_points.append((x, y))
        image_points = np.array(image_points, dtype=np.float32)

        FOCAL_FACTOR = 2.5
        focal_length = FOCAL_FACTOR * float(w)
        center = (float(w) / 2.0, float(h) / 2.0)
        camera_matrix = np.array([
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0]
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            MODEL_3D_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            sy = math.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
            singular = sy < 1e-6
            if not singular:
                rx = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                ry = math.atan2(-rotation_matrix[2, 0], sy)
                rz = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            else:
                rx = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                ry = math.atan2(-rotation_matrix[2, 0], sy)
                rz = 0.0

            roll = math.degrees(rx)
            pitch = math.degrees(ry)
            yaw = math.degrees(rz)
            yaw = max(min(yaw, 180.0), -180.0)
            pitch = max(min(pitch, 180.0), -180.0)
            roll = max(min(roll, 180.0), -180.0)
            return yaw, pitch, roll

        return _compute_fallback_head_pose(landmarks, w, h)
    except Exception:
        return _compute_fallback_head_pose(landmarks, w, h)


def calculate_slump_from_pose(pose_landmarks, w, h):
    if not pose_landmarks:
        return DUMMY_VALUE, DUMMY_VALUE
    try:
        p_left = pose_landmarks[11]
        p_right = pose_landmarks[12]
        p_nose = pose_landmarks[0]
        y_nose = p_nose.y * h
        y_shoulder_mid = ((p_left.y * h) + (p_right.y * h)) / 2.0
        d_slump = y_nose - y_shoulder_mid
        x_l = p_left.x * w; y_l = p_left.y * h
        x_r = p_right.x * w; y_r = p_right.y * h
        shoulder_angle_rad = np.arctan2(y_r - y_l, x_r - x_l)
        r_tilt = np.degrees(shoulder_angle_rad)
        return d_slump, r_tilt
    except Exception:
        return DUMMY_VALUE, DUMMY_VALUE


def run_demo(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print('Cannot open camera')
        return

    use_tasks = False
    face_landmarker = None
    pose_landmarker = None

    # Try to initialize Tasks models if available and .task files exist
    face_model_path = os.path.join('model', 'face_landmarker.task')
    pose_model_path = os.path.join('model', 'pose_landmarker_full.task')
    if TASKS_AVAILABLE and os.path.exists(face_model_path):
        try:
            base_options_face = python.BaseOptions(model_asset_path=face_model_path)
            face_options = vision.FaceLandmarkerOptions(base_options=base_options_face, running_mode=vision.RunningMode.LIVE_STREAM)
            face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
            use_tasks = True
        except Exception:
            face_landmarker = None

    # Fallback: use mediapipe solutions
    mp_face_mesh = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh, \
         mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        prev_time = time.time()
        # smoothing state for head-pose to reduce jitter
        prev_yaw = None
        prev_pitch = None
        prev_roll = None
        SMOOTH_ALPHA = 0.7  # higher = smoother/slower
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # face mesh
            face_results = face_mesh.process(rgb)
            pose_results = pose.process(rgb)

            face_disp = 0
            pose_disp = 0
            avg_ear = DUMMY_VALUE; mar = DUMMY_VALUE
            yaw = pitch = roll = DUMMY_VALUE
            d_slump = DUMMY_VALUE; r_tilt = DUMMY_VALUE

            if face_results.multi_face_landmarks:
                face_disp = 1
                landmarks = face_results.multi_face_landmarks[0].landmark
                # build pixel coordinates for calculations
                pts = [(lm.x * w, lm.y * h) for lm in landmarks]
                try:
                    left_eye_coords = [pts[i] for i in LEFT_EYE_INDICES]
                    right_eye_coords = [pts[i] for i in RIGHT_EYE_INDICES]
                    avg_ear = (eye_aspect_ratio(left_eye_coords) + eye_aspect_ratio(right_eye_coords)) / 2.0
                except Exception:
                    avg_ear = DUMMY_VALUE
                try:
                    mouth_coords = [pts[i] for i in MOUTH_INDICES]
                    mar = mouth_aspect_ratio(mouth_coords)
                except Exception:
                    mar = DUMMY_VALUE
                # head pose (COMMENTED OUT)
                # We intentionally disable the previous yaw/pitch/roll computation
                # because it's sensitive/jittery when using face-mesh landmarks.
                # If you want to restore it, uncomment the lines below.
                # try:
                #     yaw, pitch, roll = get_head_pose_from_face_mesh(landmarks, w, h)
                # except Exception:
                #     yaw, pitch, roll = _compute_fallback_head_pose(landmarks, w, h)
                yaw = pitch = roll = DUMMY_VALUE

            if pose_results.pose_landmarks:
                pose_disp = 1
                pose_lms = pose_results.pose_landmarks.landmark
                d_slump, r_tilt = calculate_slump_from_pose(pose_lms, w, h)

            # overlay
            overlay = frame.copy()
            # draw face landmarks (tesselation + contours) if present
            if face_results and getattr(face_results, 'multi_face_landmarks', None):
                try:
                    for face_landmarks in face_results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            overlay,
                            face_landmarks,
                            mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                        )
                        mp_drawing.draw_landmarks(
                            overlay,
                            face_landmarks,
                            mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                        )
                except Exception:
                    pass
            # draw pose landmarks as shoulder horizontal + nose->mid vertical
            angle_deg = DUMMY_VALUE
            cos_angle = DUMMY_VALUE
            if pose_results and getattr(pose_results, 'pose_landmarks', None):
                try:
                    plms = pose_results.pose_landmarks.landmark
                    # indices: 0 = nose, 11 = left shoulder, 12 = right shoulder
                    p_nose = plms[0]
                    p_l = plms[11]
                    p_r = plms[12]
                    x_n, y_n = int(p_nose.x * w), int(p_nose.y * h)
                    x_l, y_l = int(p_l.x * w), int(p_l.y * h)
                    x_r, y_r = int(p_r.x * w), int(p_r.y * h)
                    # horizontal shoulder line
                    cv2.line(overlay, (x_l, y_l), (x_r, y_r), (255, 0, 0), 2)
                    # midpoint of shoulders
                    mx, my = int((x_l + x_r) / 2), int((y_l + y_r) / 2)
                    cv2.circle(overlay, (mx, my), 4, (0, 0, 255), -1)
                    # vertical line from nose down to midpoint
                    cv2.line(overlay, (x_n, y_n), (mx, my), (0, 255, 255), 2)
                    # compute angle between shoulder horizontal and nose->midpoint vector
                    try:
                        # horizontal vector (shoulders)
                        hx, hy = (x_r - x_l), (y_r - y_l)
                        # vertical vector (from nose to midpoint)
                        vx, vy = (mx - x_n), (my - y_n)
                        h_norm = math.hypot(hx, hy)
                        v_norm = math.hypot(vx, vy)
                        if h_norm > 1e-6 and v_norm > 1e-6:
                            dot = hx * vx + hy * vy
                            cos_v = max(min(dot / (h_norm * v_norm), 1.0), -1.0)
                            angle_rad = math.acos(cos_v)
                            angle_deg = math.degrees(angle_rad)
                            cos_angle = cos_v
                        else:
                            angle_deg = DUMMY_VALUE
                            cos_angle = DUMMY_VALUE
                    except Exception:
                        angle_deg = DUMMY_VALUE
                        cos_angle = DUMMY_VALUE
                except Exception:
                    pass
            lines = [
                f'EAR: {avg_ear:.4f}' if avg_ear != DUMMY_VALUE else 'EAR: DUMMY',
                f'MAR: {mar:.4f}' if mar != DUMMY_VALUE else 'MAR: DUMMY',
                f'Angle: {angle_deg:.2f} deg' if angle_deg != DUMMY_VALUE else 'Angle: DUMMY',
                f'Cos: {cos_angle:.3f}' if cos_angle != DUMMY_VALUE else 'Cos: DUMMY',
                f'D_Slump: {d_slump:.2f}' if d_slump != DUMMY_VALUE else 'D_Slump: DUMMY',
                f'R_Tilt: {r_tilt:.2f}' if r_tilt != DUMMY_VALUE else 'R_Tilt: DUMMY',
                f'Face: {int(face_disp)}  Pose: {int(pose_disp)}'
            ]

            for i, txt in enumerate(lines):
                cv2.putText(overlay, txt, (10, 20 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # fps
            now = time.time()
            fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
            prev_time = now
            cv2.putText(overlay, f'FPS: {fps:.1f}', (w-120,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            cv2.imshow('Real-time Features', overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print('Starting real-time demo. Press q to quit.')
    run_demo(0)
