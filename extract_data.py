# import libraries
import mediapipe as mp
import cv2
import csv
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic

#Apply styling
mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)

# Get real time webcam feed
cap = cv2.VideoCapture(0)

# Re-run this program for each class name
class_name = 'One'

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
            print(results.multi_handedness)
            for hand_landmarks in results.multi_hand_landmarks:
                # Export coordinates
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                )

                # Extract hand landmarks to list
                hand = hand_landmarks.landmark
                hand_row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand]).flatten())

                # Append class name
                hand_row.insert(0, class_name)

                # Export to CSV
                with open('coords.csv', mode='a', newline='') as f:
                   csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                   csv_writer.writerow(hand_row)

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
