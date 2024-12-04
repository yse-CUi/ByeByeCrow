import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog, messagebox
from threading import Thread
from playsound import playsound  # 音声を再生するためにplaysoundを使用

# クラス（分類カテゴリ）
class_names = ['crow', 'chicken', 'eagle']  # 実際のカテゴリに変更してください

# データセットのパス
data_dir = 'C:/Users/230028/Desktop/images 1'  # 実際の画像データのパスに変更してください

# データを読み込む関数
def load_data(data_dir):
    images = []
    labels = []
    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128))  # 画像サイズを変更
            images.append(img)
            labels.append(idx)
    images = np.array(images)
    labels = np.array(labels)
    images = images / 255.0  # 正規化
    return images, labels

# データセットを読み込む
images, labels = load_data(data_dir)

# モデルを読み込む
model_path = 'bird_classifier_model.h5'
model = load_model(model_path)
print("保存されたモデルを読み込みました")

# GUI インターフェース
def create_gui(model):
    window = tk.Tk()
    window.title("鳥の分類器")
    window.geometry("400x200")

    # 画像選択ボタンの機能
    def select_image():
        file_path = filedialog.askopenfilename()
        if file_path:
            predicted_class = predict_image(file_path)
            messagebox.showinfo("予測結果", f"予測されたカテゴリ: {predicted_class}")

    # 画像予測関数
    def predict_image(image_path):
        try:
            # 画像を読み込む
            img = cv2.imread(image_path)
            img_resized = cv2.resize(img, (128, 128))
            img_normalized = np.expand_dims(img_resized, axis=0)  # バッチ次元を追加
            img_normalized = img_normalized / 255.0  # 正規化

            # 予測を行う
            prediction = model.predict(img_normalized)
            predicted_class = class_names[np.argmax(prediction)]

            # 画像を表示
            cv2.putText(img, f"Predicted: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Prediction", img)

            # 「カラス」を予測した場合、音声を再生するスレッドを起動
            if predicted_class == 'crow':
                play_sound_thread = Thread(target=play_sound, args=('C:/Users/230028/Desktop/images 1/crow_sound1.mp3',))
                play_sound_thread.start()

            cv2.waitKey(0)
            cv2.destroyAllWindows()

            return predicted_class
        except Exception as e:
            print(f"エラー: {e}")
            messagebox.showerror("エラー", f"エラーが発生しました: {e}")

    # 音声再生関数（playsoundを使用）
    def play_sound(sound_path):
        try:
            # 音声ファイルを再生
            playsound(sound_path)
        except Exception as e:
            print(f"音声再生エラー: {e}")

    # 画像選択ボタン
    select_button = tk.Button(window, text="画像を選択", command=select_image)
    select_button.pack(pady=50)

    # GUIを起動
    window.mainloop()

# GUIを起動
create_gui(model)