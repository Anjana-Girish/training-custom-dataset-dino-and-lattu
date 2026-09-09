import cv2
from ultralytics import YOLO
import pyttsx3
import threading
import time

# Load YOLO model
model = YOLO("my_model.pt")

# Open camera
cap = cv2.VideoCapture(0)

# Voice settings
speech_interval = 3
last_speech_time = 0
speaking = False


def speak(message):
    global speaking

    if speaking:
        return

    speaking = True

    def voice():
        global speaking

        engine = pyttsx3.init()
        engine.setProperty("rate", 150)

        engine.say(message)
        engine.runAndWait()

        engine.stop()
        speaking = False

    threading.Thread(target=voice, daemon=True).start()


while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLO
    results = model(frame, verbose=False)

    dino_detected = False
    lattu_detected = False

    # Check detections
    for result in results:

        for box in result.boxes:

            confidence = float(box.conf[0])

            if confidence < 0.50:
                continue

            class_id = int(box.cls[0])
            class_name = model.names[class_id].lower()

            if class_name == "dino":
                dino_detected = True

            elif class_name == "lattu":
                lattu_detected = True

    current_time = time.time()

    # -----------------------------
    # DINO DETECTED
    # -----------------------------

    if dino_detected:

        if current_time - last_speech_time >= speech_interval:

            speak("Danger! Dino is visible!")

            last_speech_time = current_time

    # -----------------------------
    # LATTU DETECTED
    # -----------------------------

    elif lattu_detected:

        if current_time - last_speech_time >= speech_interval:

            speak("Lattu is visible. Play time!")

            last_speech_time = current_time

    # -----------------------------
    # NOTHING DETECTED
    # -----------------------------

    else:

        last_speech_time = 0

    # Display YOLO result
    annotated_frame = results[0].plot()

    cv2.imshow(
        "Dino and Lattu Detection",
        annotated_frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()