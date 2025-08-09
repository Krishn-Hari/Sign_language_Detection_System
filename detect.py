import argparse
import json
import os
import time
import tempfile
import itertools
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from gtts import gTTS
import pygame
import string
from tensorflow import keras


DEFAULT_MODEL = "model.keras"
DEFAULT_LABELS = "labels.json"


def load_labels(labels_path: str):
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        classes = data.get("classes")
        if isinstance(classes, list) and classes:
            return classes
    # Fallback to 9 digits + 26 uppercase letters
    return [str(i) for i in range(1, 10)] + list(string.ascii_uppercase)


def load_model(model_path: str):
    if not os.path.exists(model_path):
        # Fallback to legacy H5 if Keras file is missing
        h5_fallback = "model_v2.h5"
        if os.path.exists(h5_fallback):
            return keras.models.load_model(h5_fallback)
        raise FileNotFoundError(
            f"Model not found: {model_path}. Train first or provide --model path."
        )
    return keras.models.load_model(model_path)


def extract_landmarks(img, hand_landmarks):
    img_width, img_height = img.shape[1], img.shape[0]
    landmarks = [
        [min(int(landmark.x * img_width), img_width - 1),
         min(int(landmark.y * img_height), img_height - 1)]
        for landmark in hand_landmarks.landmark
    ]
    return landmarks


def preprocess_landmarks(landmarks):
    base_x, base_y = landmarks[0][0], landmarks[0][1]
    relative_landmarks = [[x - base_x, y - base_y] for x, y in landmarks]
    flattened = list(itertools.chain.from_iterable(relative_landmarks))
    max_val = max(map(abs, flattened)) if flattened else 1.0
    normalized = [val / max_val for val in flattened] if max_val != 0 else flattened
    return normalized


def speak(text: str):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            filename = tmp.name
        tts = gTTS(text=text, lang="en")
        tts.save(filename)
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        try:
            os.remove(filename)
        except OSError:
            pass
    except Exception as e:
        print(f"TTS error: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time ISL detection")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to Keras model (.keras or .h5)")
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="Path to labels JSON")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to use")
    parser.add_argument("--conf", type=float, default=0.80, help="Confidence threshold for speech")
    parser.add_argument("--min_det", type=float, default=0.5, help="MediaPipe min detection confidence")
    parser.add_argument("--min_track", type=float, default=0.5, help="MediaPipe min tracking confidence")
    parser.add_argument("--no_audio", action="store_true", help="Disable text-to-speech")
    return parser.parse_args()


def main():
    args = parse_args()

    labels = load_labels(args.labels)
    model = load_model(args.model)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    pygame.init()
    try:
        pygame.mixer.init()
    except Exception as e:
        print(f"Audio init warning: {e}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Cannot access webcam. Check permissions.")

    last_label = None
    last_change_time = 0.0
    last_frame_time = time.time()

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=args.min_det,
        min_tracking_confidence=args.min_track,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("No frame captured from webcam.")
                continue

            now = time.time()
            fps = 1.0 / max(1e-3, (now - last_frame_time))
            last_frame_time = now

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            annotated = frame.copy()

            predicted_label = ""
            confidence = 0.0

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                landmarks = extract_landmarks(annotated, hand_landmarks)
                features = preprocess_landmarks(landmarks)
                df = pd.DataFrame([features])
                probs = model.predict(df, verbose=0)[0]
                idx = int(np.argmax(probs))
                predicted_label = labels[idx] if idx < len(labels) else str(idx)
                confidence = float(probs[idx])

                mp_drawing.draw_landmarks(
                    annotated,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

            # HUD: label, confidence, fps
            hud = []
            if predicted_label:
                hud.append(f"{predicted_label} ({confidence:.2f})")
            hud.append(f"FPS: {fps:.1f}")
            cv2.putText(
                annotated,
                " | ".join(hud),
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("ISL Detector", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

            # Speak if label changed and confidence is high
            if not args.no_audio and predicted_label and confidence >= args.conf:
                if predicted_label != last_label and (now - last_change_time) > 0.75:
                    speak(predicted_label)
                    last_label = predicted_label
                    last_change_time = now

        cap.release()
        cv2.destroyAllWindows()
        try:
            pygame.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
