import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque

# Load models đã train
model = joblib.load('drowsiness_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Khởi tạo MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Indices cho landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

# Để smooth predictions
prediction_buffer = deque(maxlen=10)

def calculate_ear(eye_landmarks):
    """Tính Eye Aspect Ratio"""
    v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    return (v1 + v2) / (2.0 * h)

def calculate_mar(mouth_landmarks):
    """Tính Mouth Aspect Ratio"""
    v1 = np.linalg.norm(mouth_landmarks[1] - mouth_landmarks[7])
    v2 = np.linalg.norm(mouth_landmarks[2] - mouth_landmarks[6])
    v3 = np.linalg.norm(mouth_landmarks[3] - mouth_landmarks[5])
    h = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[4])
    return (v1 + v2 + v3) / (3.0 * h)

def extract_features_realtime(landmarks, h, w):
    """Trích xuất features từ landmarks"""
    points = np.array([[lm.x * w, lm.y * h] for lm in landmarks.landmark])
    
    left_eye = points[LEFT_EYE]
    right_eye = points[RIGHT_EYE]
    mouth = points[MOUTH]
    
    left_ear = calculate_ear(left_eye)
    right_ear = calculate_ear(right_eye)
    avg_ear = (left_ear + right_ear) / 2.0
    mar = calculate_mar(mouth)
    eye_distance = np.linalg.norm(points[33] - points[263])
    mouth_width = np.linalg.norm(mouth[0] - mouth[4])
    
    features = np.array([
        left_ear, right_ear, avg_ear, mar,
        eye_distance, mouth_width, avg_ear / (mar + 1e-6)
    ]).reshape(1, -1)
    
    return features, avg_ear, mar

def get_smooth_prediction(pred):
    """Smooth predictions sử dụng buffer"""
    prediction_buffer.append(pred)
    if len(prediction_buffer) >= 5:
        # Lấy prediction xuất hiện nhiều nhất trong 5 frames gần nhất
        from collections import Counter
        most_common = Counter(prediction_buffer).most_common(1)[0][0]
        return most_common
    return pred

def draw_info(frame, state, ear, mar, confidence=None):
    """Vẽ thông tin lên frame"""
    h, w = frame.shape[:2]
    
    # Màu sắc theo trạng thái
    colors = {
        'awake': (0, 255, 0),      # Xanh lá
        'closed_eyes': (0, 165, 255),  # Cam
        'yawn': (0, 0, 255)         # Đỏ
    }
    
    color = colors.get(state, (255, 255, 255))
    
    # Vẽ banner phía trên
    cv2.rectangle(frame, (0, 0), (w, 80), color, -1)
    cv2.putText(frame, f"State: {state.upper()}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    # Vẽ thông tin metrics
    info_y = 120
    cv2.putText(frame, f"EAR: {ear:.3f}", (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"MAR: {mar:.3f}", (10, info_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Cảnh báo
    if state == 'closed_eyes':
        cv2.putText(frame, "CAUTION: Eyes Closed!", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    elif state == 'yawn':
        cv2.putText(frame, "WARNING: Yawning Detected!", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

def main():
    cap = cv2.VideoCapture(0)
    
    # Thiết lập độ phân giải
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Nhấn 'q' để thoát")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            
            # Vẽ face mesh (optional, có thể bỏ để tăng FPS)
            # mp_drawing.draw_landmarks(
            #     frame, landmarks, mp_face_mesh.FACEMESH_CONTOURS,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1)
            # )
            
            # Trích xuất features
            features, ear, mar = extract_features_realtime(landmarks, h, w)
            
            # Chuẩn hóa và predict
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            state = label_encoder.inverse_transform([prediction])[0]
            
            # Smooth prediction
            state = get_smooth_prediction(state)
            
            # Vẽ thông tin
            draw_info(frame, state, ear, mar)
        else:
            cv2.putText(frame, "No face detected", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('Drowsiness Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

if __name__ == "__main__":
    main()