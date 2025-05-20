import cv2
import numpy as np
import mediapipe as mp
import pyautogui
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('gesture_model.h5')  # Ensure this matches your actual model path
class_names = ['thumbs_down', 'right_click', 'move', 'left_click', 'thumbs_up']

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
screen_w, screen_h = pyautogui.size()

# Webcam input
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame to mirror the image
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert the image to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box of hand for ROI
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            x_min = int(min(x_coords) * w) - 20
            y_min = int(min(y_coords) * h) - 20
            x_max = int(max(x_coords) * w) + 20
            y_max = int(max(y_coords) * h) + 20

            # Clamp to image bounds
            x_min, y_min = max(x_min, 0), max(y_min, 0)
            x_max, y_max = min(x_max, w), min(y_max, h)

            # Crop hand ROI
            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size == 0:
                continue

            # Preprocess the hand ROI
            hand_img_resized = cv2.resize(hand_img, (128, 128))  # Must match training input
            hand_img_rgb = cv2.cvtColor(hand_img_resized, cv2.COLOR_BGR2RGB)
            hand_img_normalized = hand_img_rgb / 255.0
            input_data = np.expand_dims(hand_img_normalized, axis=0)

            # Predict gesture
            predictions = model.predict(input_data)
            pred_index = np.argmax(predictions)
            gesture = class_names[pred_index]
            confidence = predictions[0][pred_index]

            # Display the prediction
            cv2.putText(frame, f"{gesture} ({confidence:.2f})", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Virtual mouse actions
            cx = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * screen_w)
            cy = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * screen_h)

            if gesture == 'move':
                pyautogui.moveTo(cx, cy)
            elif gesture == 'left_click':
                pyautogui.click()
            elif gesture == 'right_click':
                pyautogui.click(button='right')
            elif gesture == 'thumbs_up':
                pyautogui.scroll(50)
            elif gesture == 'thumbs_down':
                pyautogui.scroll(-50)

    # Show the frame
    cv2.imshow("Virtual Mouse", frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
