import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog, messagebox
from threading import Thread
from playsound import playsound  # 音声を再生するためにplaysoundを使用
from PIL import Image, ImageTk  # 用于在Tkinter中显示图像

# クラス（分類カテゴリ）
class_names = ['crow', 'chicken', 'eagle']  # 实际的类别名称可以修改

# データセットのパス
data_dir = 'C:/Users/230028/Desktop/images 1'  # 图片数据集的路径

# データを読み込む関数
def load_data(data_dir):
    images = []
    labels = []
    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128))  # 调整图片大小
            images.append(img)
            labels.append(idx)
    images = np.array(images)
    labels = np.array(labels)
    images = images / 255.0  # 归一化
    return images, labels

# 数据集を読み込む
images, labels = load_data(data_dir)

# モデルを読み込む
model_path = 'bird_classifier_model.h5'  # 模型路径
model = load_model(model_path)
print("保存されたモデルを読み込みました")

# GUI インターフェース
def create_gui(model):
    window = tk.Tk()
    window.title("鸟的分类器")
    window.geometry("400x400")  # 调整窗口大小

    # 显示图片的函数
    def display_image_in_tkinter(image_path):
        img = cv2.imread(image_path)
        img_resized = cv2.resize(img, (128, 128))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        # 显示图像
        label = tk.Label(window, image=img_tk)
        label.image = img_tk  # 保持对图片的引用
        label.pack()

    # 画像選択ボタンの機能
    def select_image():
        file_path = filedialog.askopenfilename()
        if file_path:
            predicted_class = predict_image(file_path)
            messagebox.showinfo("预测结果", f"预测的类别: {predicted_class}")
            display_image_in_tkinter(file_path)  # 显示选择的图片

    # 画像予測関数
    def predict_image(image_path):
        try:
            # 读取图像
            img = cv2.imread(image_path)
            img_resized = cv2.resize(img, (128, 128))
            img_normalized = np.expand_dims(img_resized, axis=0)  # 增加一个批次维度
            img_normalized = img_normalized / 255.0  # 归一化

            # 进行预测
            prediction = model.predict(img_normalized)
            predicted_class = class_names[np.argmax(prediction)]

            # 在图像上添加文本
            cv2.putText(img, f"Predicted: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 如果预测为“乌鸦”，播放声音
            if predicted_class == 'crow':
                play_sound_thread = Thread(target=play_sound, args=('C:/Users/230028/Desktop/images 1/crow_sound1.mp3',))
                play_sound_thread.start()

            # 显示图像
            cv2.imshow("Prediction", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            return predicted_class
        except Exception as e:
            print(f"错误: {e}")
            messagebox.showerror("错误", f"发生错误: {e}")

    # 音声再生関数
    def play_sound(sound_path):
        try:
            playsound(sound_path)  # 播放声音
        except Exception as e:
            print(f"播放声音错误: {e}")

    # 画像選択ボタン
    select_button = tk.Button(window, text="选择图片", command=select_image)
    select_button.pack(pady=50)

    # 启动GUI
    window.mainloop()

# 启动GUI
create_gui(model)
