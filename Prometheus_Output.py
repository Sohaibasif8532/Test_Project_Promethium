import cv2
import mediapipe as mp
import urllib.request
import os

# Define the model URL and local path
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
MODEL_PATH = 'hand_landmarker.task'

def download_model_if_not_exists(url, path):
    """Downloads the MediaPipe model file if it doesn't already exist locally."""
    if not os.path.exists(path):
        print(f"Downloading {os.path.basename(path)}...")
        urllib.request.urlretrieve(url, path)
        print(f"Downloaded {os.path.basename(path)} to {path}")

def main():
    # Ensure the model is downloaded
    download_model_if_not_exists(MODEL_URL, MODEL_PATH)

    # Configure MediaPipe Hand Landmarker
    base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp.tasks.vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    # Define hand connections for drawing (manually defined as mp.solutions.hands is deprecated)
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),       # Index finger
        (9, 10), (10, 11), (11, 12),          # Middle finger
        (13, 14), (14, 15), (15, 16),         # Ring finger
        (0, 17), (17, 18), (18, 19), (19, 20), # Pinky finger
        (5, 9), (9, 13), (13, 17)             # Between fingers (MCPs)
    ]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            break

        # Flip the frame horizontally for a natural mirror effect
        frame = cv2.flip(frame, 1)
        # Convert the BGR frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect hand landmarks
        detection_result = detector.detect(mp_image)

        # Draw landmarks and connections
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # Draw connections
                for connection in HAND_CONNECTIONS:
                    start_node = hand_landmarks[connection[0]]
                    end_node = hand_landmarks[connection[1]]
                    start_point = (int(start_node.x * frame.shape[1]), int(start_node.y * frame.shape[0]))
                    end_point = (int(end_node.x * frame.shape[1]), int(end_node.y * frame.shape[0]))
                    cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

                # Draw landmarks
                for landmark in hand_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

                # Simple gesture recognition: Check for a "closed fist" (all fingers curled)
                # This is a very basic example; more complex gestures require analyzing landmark positions and angles.
                # For simplicity, let's just check if the "thumb up" (THUMB_TIP above THUMB_IP) is false and others are "down"

                # Define some landmark indices for clarity
                WRIST = 0
                THUMB_CMC = 1
                THUMB_MCP = 2
                THUMB_IP = 3
                THUMB_TIP = 4

                INDEX_FINGER_MCP = 5
                INDEX_FINGER_PIP = 6
                INDEX_FINGER_DIP = 7
                INDEX_FINGER_TIP = 8

                MIDDLE_FINGER_MCP = 9
                MIDDLE_FINGER_PIP = 10
                MIDDLE_FINGER_DIP = 11
                MIDDLE_FINGER_TIP = 12

                RING_FINGER_MCP = 13
                RING_FINGER_PIP = 14
                RING_FINGER_DIP = 15
                RING_FINGER_TIP = 16

                PINKY_MCP = 17
                PINKY_PIP = 18
                PINKY_DIP = 19
                PINKY_TIP = 20

                # A very crude "fist" detection: if fingertips are significantly below their PIP joints (Y-axis check)
                is_fist = False
                if (hand_landmarks[INDEX_FINGER_TIP].y > hand_landmarks[INDEX_FINGER_PIP].y and
                    hand_landmarks[MIDDLE_FINGER_TIP].y > hand_landmarks[MIDDLE_FINGER_PIP].y and
                    hand_landmarks[RING_FINGER_TIP].y > hand_landmarks[RING_FINGER_PIP].y and
                    hand_landmarks[PINKY_TIP].y > hand_landmarks[PINKY_PIP].y and
                    # Additionally, ensure thumb is not extended relative to its IP joint on the x-axis
                    hand_landmarks[THUMB_TIP].x < hand_landmarks[THUMB_IP].x):
                    is_fist = True


                # A very crude "open hand" detection: if fingertips are significantly above their MCP joints (Y-axis check)
                is_open_hand = False
                if (hand_landmarks[INDEX_FINGER_TIP].y < hand_landmarks[INDEX_FINGER_MCP].y and
                    hand_landmarks[MIDDLE_FINGER_TIP].y < hand_landmarks[MIDDLE_FINGER_MCP].y and
                    hand_landmarks[RING_FINGER_TIP].y < hand_landmarks[RING_FINGER_MCP].y and
                    hand_landmarks[PINKY_TIP].y < hand_landmarks[PINKY_MCP].y and
                    # Additionally, ensure thumb is extended relative to its IP joint on the x-axis
                    hand_landmarks[THUMB_TIP].x > hand_landmarks[THUMB_IP].x):
                    is_open_hand = True


                # Display gesture
                gesture_text = "No gesture"
                if is_fist:
                    gesture_text = "Fist"
                elif is_open_hand:
                    gesture_text = "Open Hand"

                cv2.putText(frame, gesture_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


        # Display the frame
        cv2.imshow('Hand Gesture AI', frame)

        # Break the loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    detector.close()

if __name__ == '__main__':
    main()