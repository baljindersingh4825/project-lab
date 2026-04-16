import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import sqlite3
import winsound
from tensorflow.keras.models import load_model

# Load CNN Model
model = load_model("drowsiness_cnn.h5")

# Database setup
conn = sqlite3.connect("drowsiness.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    status TEXT
)
""")
conn.commit()

# Camera
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not working ")
    exit()

# MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]

EYE_CLOSE_TIME = 3

eye_closed_start = None
alarm_running = False


# 🔊 CONTINUOUS BEEP (BEST)
def alarm_sound():
    global alarm_running
    while alarm_running:
        winsound.Beep(2000, 500)  # high pitch
        winsound.Beep(1500, 500)  # low pitch


# 💾 SAVE EVENT
def save_event(status):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO events (timestamp, status) VALUES (?, ?)", (timestamp, status))
    conn.commit()


# 👁️ EXTRACT EYE
def get_eye_region(frame, landmarks, indices, w, h):
    points = []
    for i in indices:
        x = int(landmarks.landmark[i].x * w)
        y = int(landmarks.landmark[i].y * h)
        points.append((x, y))

    x_min = min(p[0] for p in points)
    x_max = max(p[0] for p in points)
    y_min = min(p[1] for p in points)
    y_max = max(p[1] for p in points)

    eye = frame[y_min:y_max, x_min:x_max]

    if eye.size == 0:
        return None

    eye = cv2.resize(eye, (24, 24))
    eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    eye = eye / 255.0
    eye = eye.reshape(1, 24, 24, 1)

    return eye


# 🚀 MAIN LOOP
while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    status = "AWAKE"

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            h, w, _ = frame.shape

            left_eye = get_eye_region(frame, face_landmarks, LEFT_EYE, w, h)
            right_eye = get_eye_region(frame, face_landmarks, RIGHT_EYE, w, h)

            if left_eye is None or right_eye is None:
                continue

            left_pred = model.predict(left_eye, verbose=0)[0][0]
            right_pred = model.predict(right_eye, verbose=0)[0][0]

            # 🔥 IMPROVED LOGIC
            avg_pred = (left_pred + right_pred) / 2
            eyes_closed = avg_pred < 0.6

            if eyes_closed:

                if eye_closed_start is None:
                    eye_closed_start = time.time()

                elapsed = time.time() - eye_closed_start

                if elapsed >= EYE_CLOSE_TIME:

                    status = "DROWSY"

                    cv2.putText(frame,
                                "DROWSINESS ALERT!",
                                (40,100),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2,
                                (0,0,255),
                                3)

                    # 🔊 START ALARM
                    if not alarm_running:
                        alarm_running = True
                        threading.Thread(target=alarm_sound, daemon=True).start()

                    save_event("DROWSY")

            else:
                # 🔇 STOP ALARM
                eye_closed_start = None
                alarm_running = False
                status = "AWAKE"

            cv2.putText(frame,
                        f"Status: {status}",
                        (20,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,255,0),
                        2)

    cv2.imshow("AI Driver Monitoring System", frame)

    # ✅ EXIT CONTROLS
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):   # ESC or Q
        break


# CLEANUP
alarm_running = False
cap.release()
conn.close()
cv2.destroyAllWindows()