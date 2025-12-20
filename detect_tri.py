import time
import math
import cv2
import numpy as np
import os
import joblib
import tensorflow as tf
from scipy.spatial import distance as dist
from collections import deque
from collections import Counter

try:
    import mediapipe as mp
    TASKS_AVAILABLE = True
except Exception:
    import mediapipe as mp
    TASKS_AVAILABLE = False

# --- Constants & Config ---
MODEL_PATH = './model/best_drowsiness_model.keras' 
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

# --- CONFIG FOR SENSITIVITY ---
# --- CẤU HÌNH ĐỘ TRỄ ---
# Giả sử camera chạy 30 FPS.
# Muốn delay 2 giây mới báo Sleep -> Cần window size khoảng 60 frames.
# Tuy nhiên để UI mượt hơn, ta để window tầm 15-30 frame là vừa đủ lọc nhiễu blink.

WINDOW_SIZE = 20
THRESHOLD_RATIO = 0.6 # Trạng thái phải chiếm 60% trong window mới được coi là thật
FAST_AWAKE_FRAMES = 5 #  Chỉ cần 5 frame (0.15s) AWAKE liên tiếp để phá bỏ trạng thái ngủ

# --- Load Model & Scaler ---
print(f"Loading Scaler from {SCALER_PATH}...")
try:
    with open(SCALER_PATH, 'rb') as f: # Sửa lại cách load pickle chuẩn
        scaler = joblib.load(SCALER_PATH)
        print("Scaler Loaded!")
except Exception as e:
    print(f"FATAL ERROR: Could not load scaler.pkl. {e}")
    exit()

print(f"Loading Model from {MODEL_PATH}...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model Loaded!")
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

    # Cấu hình Camera Buffer nhỏ lại để tránh lag hình ảnh
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    mp_face_mesh = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles 
    
    # --- WAITNG QUEUE FOR PREDICTION) ---
    prediction_history = deque(maxlen=WINDOW_SIZE)
    
    # Biến lưu trạng thái hiển thị cuối cùng (để chống rung UI)
    final_label = "INITIALIZING"
    final_color = (200, 200, 200)
    final_conf = 0.0

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

            # --- 3. NEW PREDICTION LOGIC (SOOTHING) ---
            if face_disp:
                # Prediction liên tục (no delay)
                input_features = np.array([[ear, mar, pitch, yaw, roll, d_slump, r_tilt]])
                scaled_features = scaler.transform(input_features)
                preds = model.predict(scaled_features, verbose=0)[0]
                idx = np.argmax(preds)
                predicted_label = LABELS[idx]

                # Let result in the queue
                prediction_history.append(predicted_label)

                # --- THUẬT TOÁN ĐA SỐ NÂNG CẤP (UPDATED MAJORITY VOTING) ---

                # [QUY TẮC MỚI 1] CƠ CHẾ THỨC KHẨN CẤP (FAST AWAKE)
                # Nếu 5 frame gần nhất ĐỀU là AWAKE -> Báo thức ngay (bỏ qua buffer cũ)
                recent_history = list(prediction_history)[-FAST_AWAKE_FRAMES:]
                if len(recent_history) == FAST_AWAKE_FRAMES and all (label == "AWAKE" for label in recent_history):
                    final_label = "AWAKE"
                    final_conf = preds[0]
                    prediction_history.clear() # Xóa buffer ngủ để reset

                # [QUY TẮC MỚI 2] XỬ LÝ BUFFER
                elif len(prediction_history) == WINDOW_SIZE:
                    # Buffer đã đầy, dùng Majority Voting
                    # Đếm xem trạng thái nào xuất hiện nhiều nhất trong 30 frame vừa qua
                    count = Counter(prediction_history)
                    most_common_label, frequency = count.most_common(1)[0]

                    #  trạng thái này chiếm > 70% lịch sử (THRESHOLD_RATIO) -> Chốt 
                    if frequency / WINDOW_SIZE >= THRESHOLD_RATIO:
                        final_label = most_common_label
                        final_conf = preds[idx] # Lấy conf hiện tại
                
                # [QUY TẮC MỚI 3] WARM-UP (KHỞI ĐỘNG)
                # Khi mới detect lại mặt (buffer chưa đầy), hiển thị tạm kết quả RAW
                # để người dùng không thấy bị đơ.
                elif len(prediction_history) < WINDOW_SIZE:
                        if final_label == "NO FACE DETECTION": # Chỉ override nếu trước đó là mất mặt
                            final_label = predicted_label + "Calibrating..."
                            final_conf = preds[idx]

                # Map lại màu sắc từ Label text
                if "AWAKE" in final_label: final_color = COLORS[0]
                elif "SLEEP" in final_label: final_color = COLORS[1]
                elif "YAWN" in final_label: final_color = COLORS[2]
                else: final_color = (200, 200, 200)

            else:
                # [QUY TẮC MỚI 4] MẤT MẶT TỨC THÌ
                # Không chờ đợi, báo ngay lập tức
                prediction_history.clear()
                final_label = "NO FACE DETECTION"
                final_color = (100, 100, 100)
                final_conf = 0.0

            # --- 4. DISPLAY ---
            # Thanh trạng thái buffer (Visual debug để bạn thấy độ trễ)
            # Vẽ một thanh loading nhỏ thể hiện buffer đang đầy bao nhiêu
            cv2.rectangle(overlay, (5, 5), (400, 150), (0, 0, 0), -1)
            output_frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)


            # Big Result on Top
            cv2.putText(output_frame, f"{final_label}", (20, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, final_color, 3)
            
            # Show confidence
            if final_conf > 0: 
                cv2.putText(output_frame, f"Confidence: {final_conf*100:.1f}%", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                
            # Vẽ thanh buffer
            buffer_len = len(prediction_history)
            bar_color = (0, 255, 255) if buffer_len < WINDOW_SIZE else (0, 255, 0)
            cv2.rectangle(output_frame, (20, 120), (20 + buffer_len * 5, 130), bar_color, -1)
            cv2.putText(output_frame, "Buffer Stability", (20 + WINDOW_SIZE*5 + 10, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

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
                f"Raw Pred: {LABELS[idx] if face_disp else 'None'}", # Dự đoán thô (nhảy loạn xạ)
                f"Stable:   {final_label}" # Dự đoán đã lọc (đầm hơn)
            ]
            
            # Draw stats
            for i, line in enumerate(lines):
                y_pos = h - 300 + (i* 25)
                cv2.putText(output_frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 2)
                
            cv2.imshow("Smoothed Drowsiness System", output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_demo()