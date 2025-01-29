import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from threading import Thread
from playsound import playsound
import time

# クラス（分類カテゴリ）
class_names = ['crow', 'chicken', 'eagle']  # 实际的分类标签

# 模型路径
model_path = os.path.join(os.getcwd(), 'bird_classifier_model.h5')  # 模型的相对路径
model = load_model(model_path)
print("保存されたモデルを読み込みました")

# 音声播放控制变量
last_played_time = 0  # 上次播放的时间
play_interval = 2.0   # 音频播放间隔（秒）
current_class = None  # 当前预测类别
is_playing = False    # 是否正在播放音频的标志

# 音频播放函数（带时间间隔控制）
def play_sound_with_interval(sound_path):
    global last_played_time, is_playing
    current_time = time.time()

    # 控制音频播放的间隔
    if not is_playing and current_time - last_played_time > play_interval:
        is_playing = True
        playsound(sound_path)  # 播放音频
        last_played_time = current_time  # 更新最后播放的时间
        is_playing = False  # 播放结束后重置标志

# 音频播放函数（如果识别为“crow”且置信度超过50%播放音频）
def play_sound(predicted_class, confidence, sound_path):
    global current_class
    print(confidence, predicted_class)
    if predicted_class == "crow" and confidence > 0.5:  # 确保只在置信度超过0.5时播放
        print('sound play')
        current_class = predicted_class  # 更新当前预测类别
        threaded_play_sound(sound_path)  # 播放声音

# 音频播放线程
def threaded_play_sound(sound_path):
    """通过线程播放音频，避免阻塞"""
    sound_thread = Thread(target=play_sound_with_interval, args=(sound_path,))
    sound_thread.start()

# Web摄像头实时视频分类
def classify_video():
    # 打开Web摄像头
    cap = cv2.VideoCapture(0)  # 设备ID为0的默认摄像头
    if not cap.isOpened():
        print("カメラを起動できませんでした")
        return

    print("カメラを起動しました。終了するにはウィンドウを閉じてください。")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("カメラ映像を取得できませんでした")
            break

        # 处理每一帧图像
        img_resized = cv2.resize(frame, (128, 128))  # 将图像大小调整为128x128
        img_normalized = np.expand_dims(img_resized, axis=0)  # 增加批次维度
        img_normalized = img_normalized / 255.0  # 归一化处理

        # 预测
        prediction = model.predict(img_normalized)
        predicted_class_idx = np.argmax(prediction)
        predicted_class = class_names[predicted_class_idx]
        confidence = prediction[0][predicted_class_idx]

        # 在图像上绘制预测结果
        label = f"{predicted_class} ({confidence * 100:.2f}%)"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 如果预测为“crow”且置信度超过50%，播放音频
        if predicted_class == 'crow' and confidence > 0.5:
            sound_path = os.path.join(os.getcwd(), 'crow_sound1.mp3')  # 音频文件路径
            play_sound_thread = Thread(target=play_sound_with_interval, args=(sound_path,))
            play_sound_thread.start()

        # 显示视频帧
        cv2.imshow("鳥の分類", frame)

        # 按's'键退出
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    # 释放摄像头和窗口资源
    cap.release()
    cv2.destroyAllWindows()

# 启动视频分类
classify_video()

