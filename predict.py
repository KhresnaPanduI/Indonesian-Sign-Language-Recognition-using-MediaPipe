import pickle
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd

# Load Random Forest model
#with open('LogisticRegression.pkl', 'rb') as f:
#    model = pickle.load(f)
import joblib
model = joblib.load('LogisticRegression.pkl')

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic

# Apply styling
mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)

# Get real time webcam feed
# 0 for default laptop webcam
# 2 for dbE C200
cap = cv2.VideoCapture(0)


with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Export coordinates
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                )

                # Extract hand landmarks to list
                hand = hand_landmarks.landmark
                hand_row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand]).flatten())

                # Make predictions
                X = pd.DataFrame([hand_row])
                sign_language_class = model.predict(X)[0]
                sign_language_prob = model.predict_proba(X)[0]
                print(sign_language_class, sign_language_prob)

                # Get status box
                cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)

                # Display class
                cv2.putText(image, 'CLASS', (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, sign_language_class.split(' ')[0], (90, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, -1, (255, 255, 255), 2, cv2.LINE_AA)

                # Display probability
                cv2.putText(image, 'PROB', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(round(sign_language_prob[np.argmax(sign_language_prob)], 2)), (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 2, cv2.LINE_AA))

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
