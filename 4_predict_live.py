import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import threading
import os

# --- CẤU HÌNH ---
MODEL_FILE = 'model.pkl'       # Tên file model đã train
ALERT_TIME_THRESHOLD = 3       # Thời gian (giây) để kích hoạt cảnh báo
COOLDOWN_TIME = 3              # Thời gian nghỉ giữa các lần kêu
RESIZE_WIDTH = 640             # Giảm độ phân giải để chạy nhanh hơn (640 hoặc 480)

# Danh sách 20 điểm mốc (PHẢI KHỚP CHÍNH XÁC VỚI FILE 1_COLLECT_DATA.PY)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 81, 178, 13, 14, 17, 402]
LANDMARK_INDICES = LEFT_EYE + RIGHT_EYE + MOUTH

# --- BIẾN TOÀN CỤC (Dùng để chia sẻ giữa các luồng) ---
current_prediction = "Waiting..."     
prediction_lock = threading.Lock() 

# Biến logic cảnh báo
start_drowsy_time = None
last_alert_time = 0

# --- 1. TẢI MODEL AI ---
if not os.path.exists(MODEL_FILE):
    print(f"[LỖI] Không tìm thấy file '{MODEL_FILE}'.")
    print("Hãy chắc chắn bạn đã chạy train_model.ipynb và lưu model thành công.")
    exit()

try:
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    print(f"[INFO] Đã tải Model thành công: {MODEL_FILE}")
except Exception as e:
    print(f"[LỖI] Không thể tải model: {e}")
    exit()

# --- 2. KHỞI TẠO MEDIAPIPE ---
mp_face_mesh = mp.solutions.face_mesh
# refine_landmarks=False để chạy nhanh hơn. Đổi thành True nếu máy mạnh.
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- HÀM TRÍCH XUẤT TỌA ĐỘ ---
def get_landmarks(results):
    features = []
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        for i in LANDMARK_INDICES:
            lm = landmarks[i]
            features.extend([lm.x, lm.y, lm.z])
    return features

# --- HÀM DỰ ĐOÁN (CHẠY TRÊN LUỒNG RIÊNG) ---
def predict_worker(features):
    global current_prediction
    try:
        # Chuyển đổi dữ liệu cho đúng định dạng của Model (1 hàng, 60 cột)
        X = np.array(features).reshape(1, -1)
        
        # Dự đoán
        pred = model.predict(X)[0]
        
        # Cập nhật kết quả an toàn
        with prediction_lock:
            current_prediction = pred
    except Exception as e:
        print(f"Lỗi dự đoán: {e}")

# --- HÀM MAIN ---
def main():
    global start_drowsy_time, last_alert_time, current_prediction

    cap = cv2.VideoCapture(0)
    
    # Tính toán FPS
    pTime = 0
    
    print("[INFO] Camera đang bật... Nhấn 'ESC' hoặc 'q' để thoát.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. Giảm kích thước ảnh để xử lý nhanh hơn
        h, w, c = frame.shape
        if w > RESIZE_WIDTH:
            scale = RESIZE_WIDTH / w
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))

        # 2. Xử lý MediaPipe
        frame.flags.writeable = False
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        frame.flags.writeable = True
        
        # 3. Lấy tọa độ khuôn mặt
        features = get_landmarks(results)
        
        # 4. Logic Đa luồng: Chỉ chạy AI nếu tìm thấy mặt
        if len(features) == 60:
            # Kiểm tra xem có luồng AI nào đang chạy không
            active_threads = threading.enumerate()
            ai_running = any(t.name == "AI_Worker" for t in active_threads)
            
            # Nếu không có luồng AI nào chạy, tạo luồng mới
            if not ai_running:
                t = threading.Thread(target=predict_worker, args=(features,), name="AI_Worker")
                t.daemon = True # Tự động tắt khi chương trình tắt
                t.start()

            # Lấy kết quả mới nhất
            with prediction_lock:
                status = current_prediction

            # --- HIỂN THỊ KẾT QUẢ ---
            is_drowsy = (status == 'closed_eyes') or (status == 'yawn')
            
            # Chọn màu
            if status == 'alert': 
                color = (0, 255, 0) # Xanh lá
            elif status == 'closed_eyes': 
                color = (0, 0, 255) # Đỏ
            elif status == 'yawn':
                color = (0, 165, 255) # Cam
            else:
                color = (200, 200, 200) # Xám

            # Vẽ trạng thái
            cv2.putText(frame, f"STATUS: {status.upper()}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # --- LOGIC CẢNH BÁO ---
            current_time = time.time()
            if is_drowsy:
                if start_drowsy_time is None:
                    start_drowsy_time = current_time
                
                duration = current_time - start_drowsy_time
                
                # Nếu buồn ngủ quá ngưỡng thời gian
                if duration > ALERT_TIME_THRESHOLD:
                    msg = f"!!! CANH BAO ({int(duration)}s) !!!"
                    cv2.putText(frame, msg, (20, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # In ra terminal (hoặc phát âm thanh ở đây)
                    if current_time - last_alert_time > COOLDOWN_TIME:
                        print(f"[ALARM] Tài xế buồn ngủ! Trạng thái: {status}")
                        # winsound.Beep(2500, 1000) # Nếu dùng Windows có thể bỏ comment dòng này
                        last_alert_time = current_time
            else:
                start_drowsy_time = None # Reset nếu tỉnh lại

        else:
            cv2.putText(frame, "KHONG TIM THAY MAT", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Hiển thị FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1] - 100, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Show frame
        cv2.imshow('Drowsiness AI System', frame)
        
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'): # Nhấn ESC hoặc q để thoát
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()