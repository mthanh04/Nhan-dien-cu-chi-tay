import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import time

# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Cấu hình thông số video
fps = 30  # Số khung hình mỗi giây
duration = 3  # Thời gian thu (giây)
total_frames = fps * duration  # Tổng số frame cần ghi (90)

# Mở webcam
cap = cv2.VideoCapture(0)
frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

if not cap.isOpened():
    print("❌ Không thể mở webcam!")
    exit()

# Nhập tên cử chỉ
gesture_name = "Stop"
dataset_path = f"D:/HAND-GESTURE/hand-gesture-mediapipe/DataSet/Stop"
video_path = os.path.join(dataset_path, "Videos")
csv_path = os.path.join(dataset_path, f"Stop80.csv")

# Tạo thư mục nếu chưa có
os.makedirs(video_path, exist_ok=True)

# Đếm số video đã có để đặt tên file
video_index = len(os.listdir(video_path))
video_filename = os.path.join(video_path, f"{gesture_name}_{video_index}.mp4")

# Ghi video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

print(f"\n🎥 Bắt đầu thu video {video_index + 1} ({gesture_name}) trong 3 giây...")

keypoints_list = []
start_time = time.time()

while time.time() - start_time < duration:  # Tự dừng sau 3 giây
    ret, frame = cap.read()
    if not ret:
        print("❌ Không thể đọc từ camera.")
        break

    frame = cv2.flip(frame, 1)  # Lật ảnh để đúng chiều

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    keypoints = []
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Trích xuất tọa độ keypoints (21 điểm * 3 tọa độ = 63)
            keypoints = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]

    # Lưu keypoints vào danh sách nếu đủ 63 điểm
    if len(keypoints) == 63:
        keypoints_list.append(keypoints)

    # Vẽ thời gian còn lại lên màn hình
    elapsed_time = time.time() - start_time
    remaining_time = max(0, duration - elapsed_time)
    cv2.putText(frame, f"Time: {remaining_time:.1f}s", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Ghi video
    out.write(frame)
    cv2.imshow("Recording", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("⏹ Dừng sớm do nhấn 'q'")
        break

out.release()

# Ghi tọa độ vào CSV
with open(csv_path, mode="a", newline="") as file:
    writer = csv.writer(file)
    for keypoints in keypoints_list:
        writer.writerow(keypoints + [gesture_name])

print(f"Video đã lưu tại {video_filename}")
cap.release()
cv2.destroyAllWindows()
print("Hoàn thành thu thập dữ liệu!")
