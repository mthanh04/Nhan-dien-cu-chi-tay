import cv2
import numpy as np
import tensorflow as tf
import pyautogui
import time
import os
import pygetwindow as gw  # Để đặt cửa sổ luôn trên cùng
from mediapipe import solutions

# Tải mô hình GAFormer đã huấn luyện
model = tf.keras.models.load_model("gaformer_hand_gesture.h5")

# Danh sách nhãn cử chỉ tay
labels = ["Call", "Finger_Gun", "Left", "OK", "Open", "Right", "Stop", "Thumbs_Up"]

# Khởi tạo MediaPipe Hands
mp_hands = solutions.hands
mp_draw = solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Biến lưu thời gian thao tác cuối cùng
last_action_time = 0  

def to_gaf(data):
    """ Chuyển đổi tọa độ sang Gramian Angular Field (GAF) """
    gaf_data = []
    for sample in data:
        norm_data = (sample - np.min(sample)) / (np.max(sample) - np.min(sample) + 1e-8)
        norm_data = np.array(norm_data, dtype=np.float32)
        norm_data = np.clip(norm_data, -1, 1)

        gaf = np.arccos(norm_data)
        gaf_data.append(np.cos(gaf @ gaf.T))  # Ma trận Gramian

    return np.array(gaf_data)

def preprocess_frame(frame):
    """ Xử lý ảnh từ webcam để lấy keypoints từ bàn tay """
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.append([lm.x, lm.y, lm.z])

            keypoints = np.array(keypoints).reshape(1, 21, 3)  # Định dạng đúng
            gaf_data = to_gaf(keypoints)
            gaf_data = gaf_data[..., np.newaxis]  # Thêm chiều kênh

            # Dự đoán cử chỉ
            prediction = model.predict(gaf_data)
            predicted_label = labels[np.argmax(prediction)]
            return predicted_label, hand_landmarks
    return None, None

def control_powerpoint(gesture):
    """ Điều khiển PowerPoint bằng cử chỉ tay """
    global last_action_time
    current_time = time.time()

    if current_time - last_action_time >= 2:  # Giảm độ nhạy xuống 2 giây để tránh lỗi nhận diện liên tục
        last_action_time = current_time  # Cập nhật thời gian thực hiện lệnh

        if gesture == "Open":
           if "POWERPNT.EXE" not in os.popen('tasklist').read(): # Kiểm tra xem PowerPoint đã mở chưa
            os.system("start powerpnt")  # Mở PowerPoint nếu chưa mở
            time.sleep(3)  # Chờ PowerPoint khởi động

            pyautogui.press("f5")  # Bắt đầu trình chiếu

        elif gesture == "Right":
            pyautogui.press("right")  # Chuyển slide tiếp theo

        elif gesture == "Left":
            pyautogui.press("left")  # Quay lại slide trước

        elif gesture == "Finger_Gun": # Chuyển đến slide đầu tiên
            pyautogui.press("home")

        elif gesture == "Thumbs_Up": # Chuyển đến slide đầu tiên
            pyautogui.press("end")

        elif gesture == "Stop":
            pyautogui.press("esc")  # Thoát trình chiếu

        elif gesture == "Call":
            pyautogui.press("b")  # Tạm dừng trình chiếu (chuyển màn hình đen)

        elif gesture == "Ok":
            pyautogui.press("n")  # Tiếp tục trình chiếu từ slide hiện tại

# Mở camera và chạy chương trình
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Lật ngược chiều ngang

    gesture, hand_landmarks = preprocess_frame(frame)

    if hand_landmarks:
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if gesture:
        cv2.putText(frame, f"Gesture: {gesture}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        control_powerpoint(gesture)  # Điều khiển PowerPoint

    cv2.imshow("Hand Gesture Control for PowerPoint", frame)

    # Đặt cửa sổ webcam luôn trên màn hình
    try:
        win = gw.getWindowsWithTitle("Hand Gesture Control for PowerPoint")[0]
        win.alwaysOnTop = True  # Đặt chế độ "Always on Top"
    except IndexError:
        pass  # Nếu không tìm thấy cửa sổ, bỏ qua lỗi

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Đã dừng nhận diện cử chỉ.")
