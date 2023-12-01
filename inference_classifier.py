import pickle

import cv2
import mediapipe as mp
import numpy as np
import msvcrt
import threading
import keyboard
import os


output = ""


def get_video_input():
    model_dict = pickle.load(open('./model.p','rb'))
    model = model_dict['model']

    # max_length = max(len(item) for item in model['data'])
    # data = np.array([item + [0] * (max_length - len(item)) for item in model['data']])



    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    labels_dict = {0: 'hi how are u', 1: 'See you later, thanks for visiting', 2: 'What kinds of items are there?', 3: 'Do you take credit cards?', 4: 'When do I get my delivery?'}
    while True:

        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)])

            # print(type(prediction))

            output = labels_dict[int(prediction[0])]

            print(labels_dict[int(prediction[0])])

            predicted_character = labels_dict[int(prediction[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

        cv2.imshow('frame', frame)
        if keyboard.is_pressed('enter'):
            break
        cv2.waitKey(20)

    cap.release()
    cv2.destroyAllWindows()
    return output

# Create a thread for the video input
video_thread = threading.Thread(target=get_video_input)
video_thread.start()

# Wait for the thread to finish (optional)
video_thread.join()