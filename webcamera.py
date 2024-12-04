import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from threading import Thread
from playsound import playsound  # 音声を再生するためにplaysoundを使用

# クラス（分類カテゴリ）
class_names = ['crow', 'chicken', 'eagle']  # 実際のカテゴリに変更してください

# モデルを読み込む
model_path = os.path.join(os.getcwd(), 'bird_classifier_model.h5')  # モデルの相対パス
model = load_model(model_path)
print("保存されたモデルを読み込みました")

# 音声再生関数（playsoundを使用）
def play_sound(sound_path):
    try:
        playsound(sound_path)
    except Exception as e:
        print(f"音声再生エラー: {e}")

# Webカメラを使ったリアルタイム映像処理
def classify_video():
    # Webカメラを初期化
    cap = cv2.VideoCapture(0)  # カメラデバイスを開く（デバイスID 0）
    if not cap.isOpened():
        print("カメラを起動できませんでした")
        return

    print("カメラを起動しました。終了するにはウィンドウを閉じてください。")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("カメラ映像を取得できませんでした")
            break

        # 映像フレームを前処理
        img_resized = cv2.resize(frame, (128, 128))  # モデルに合わせてサイズ変更
        img_normalized = np.expand_dims(img_resized, axis=0)  # バッチ次元を追加
        img_normalized = img_normalized / 255.0  # 正規化

        # 予測を行う
        prediction = model.predict(img_normalized)
        predicted_class_idx = np.argmax(prediction)
        predicted_class = class_names[predicted_class_idx]
        confidence = prediction[0][predicted_class_idx]

        # 映像に予測結果を描画
        label = f"{predicted_class} ({confidence * 100:.2f}%)"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # カラスの場合、音声アラートを再生
        if predicted_class == 'crow' and confidence > 0.8:  # 確信度が80%以上
            sound_path = os.path.join(os.getcwd(), 'crow_sound1.mp3')  # 音声の相対パス
            play_sound_thread = Thread(target=play_sound, args=(sound_path,))
            play_sound_thread.start()

        # 映像を表示
        cv2.imshow("鳥の分類", frame)

        # キー入力で終了
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q'キーで終了
            break

    # カメラとウィンドウを解放
    cap.release()
    cv2.destroyAllWindows()

# Webカメラ映像の分類を開始
classify_video()