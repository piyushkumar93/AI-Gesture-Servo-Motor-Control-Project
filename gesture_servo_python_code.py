import cv2
import numpy as np
import serial
import time
import tensorflow as tf

# ========== CONFIGURATION ==========
SERIAL_PORT = 'COM7'  # Change this
SERIAL_BAUDRATE = 9600
MODEL_PATH = "model.tflite"
LABELS_PATH = "labels.txt"
CONFIDENCE_THRESHOLD = 0.85
SERIAL_COOLDOWN = 0.7  # Only send serial once every 700ms
# ===================================

# Load labels
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# Setup serial connection
arduino = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE)
time.sleep(2)

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open webcam")
    exit()

prev_label = ""
last_sent_time = 0

print("[INFO] Starting real-time gesture detection...")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Resize and normalize frame
    img = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(img.astype(np.float32) / 255.0, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    max_index = int(np.argmax(output_data))
    label = labels[max_index]
    confidence = float(output_data[max_index])

    # Show debug
    print(f"[DEBUG] Detected: {label}, Confidence: {confidence:.2f}")

    # Serial cooldown check
    current_time = time.time()
    if confidence > CONFIDENCE_THRESHOLD and label != prev_label and current_time - last_sent_time > SERIAL_COOLDOWN:
        arduino.write((label + "\n").encode())
        print(f"[SERIAL] Sent to Arduino: {label}")
        prev_label = label
        last_sent_time = current_time

    # Show webcam with overlay
    cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Gesture Detection", frame)
    cv2.moveWindow("Gesture Detection", 100, 100)  # Ensure it's visible

    # Quit if 'q' pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
arduino.close()
