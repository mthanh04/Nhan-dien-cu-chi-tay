import numpy as np
import pandas as pd
import tensorflow as tf
import glob
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Bước 1: Gộp tất cả file CSV và kiểm tra dữ liệu
labels = ["Call", "Finger_Gun", "Left", "OK", "Open", "Right", "Stop", "Thumbs_Up"]
data_list, label_list = [], []

for label_index, label in enumerate(labels):
    csv_files = glob.glob(f"D:/HAND-GESTURE/hand-gesture-mediapipe/DataSet/{label}/*.csv")
    
    for csv_file in csv_files:
        data = pd.read_csv(csv_file, header=None).values
        
        # Nếu dữ liệu có 64 cột, loại bỏ cột cuối cùng
        if data.shape[1] == 64:
            data = data[:, :63]
        
        # Kiểm tra lại kích thước sau khi sửa
        if data.shape[1] == 63:
            data_list.append(data)
            label_list.append(np.full((len(data),), label_index))
        else:
            print(f"⚠️ File lỗi (sau khi cắt cột): {csv_file} - Kích thước {data.shape}")

# Chuyển dữ liệu thành mảng numpy
X = np.vstack(data_list)  # Ghép tất cả dữ liệu
y = np.concatenate(label_list)  # Ghép tất cả nhãn
X = X.reshape(-1, 21, 3)  # (samples, keypoints, tọa độ)

# Bước 2: Chuyển đổi sang Gramian Angular Field (GAF)
def to_gaf(data):
    gaf_data = []
    for sample in data:
        # Chuẩn hóa về khoảng [0, 1]
        norm_data = (sample - np.min(sample)) / (np.max(sample) - np.min(sample) + 1e-8)
        norm_data = np.array(norm_data, dtype=np.float32)  # Chuyển thành NumPy array
        norm_data = np.clip(norm_data, -1, 1)  # Giới hạn giá trị hợp lệ

        gaf = np.arccos(norm_data)  # Chuyển đổi sang góc
        gaf_data.append(np.cos(gaf @ gaf.T))  # Ma trận Gramian

    return np.array(gaf_data)


# Chuyển đổi dữ liệu
X_gaf = to_gaf(X)
X_gaf = X_gaf[..., np.newaxis]  # Thêm chiều kênh (samples, 21, 21, 1)


# Bước 3: Chia tập huấn luyện / kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_gaf, y, test_size=0.1, random_state=42)

# Bước 4: Xây dựng mô hình GAFormer
def build_gaformer(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Transformer Encoder
    x = layers.Reshape((-1, x.shape[-1]))(x)
    attn_output = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = layers.Add()([x, attn_output])
    x = layers.LayerNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs)

# Khởi tạo và biên dịch mô hình
model = build_gaformer((21, 21, 1), len(labels))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Bước 5: Huấn luyện mô hình
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=16)

# Lưu mô hình
model.save("gaformer_hand_gesture.h5")
print("Mô hình đã lưu thành công!")

# Bước 6: Vẽ biểu đồ Accuracy & Loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
b
plt.show()
