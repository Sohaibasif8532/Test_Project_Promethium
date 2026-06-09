import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request

# Define constants for model download and file path
MODEL_DIR = "models"
MODEL_FILE_NAME = "hand_landmarker.task"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE_NAME)
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# Define hand landmark connections for drawing
# These are derived from MediaPipe's standard hand connections
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky finger
    (5, 9), (9, 13), (13, 17) # Palm connections
]

def download_model_if_not_exists():
    """Downloads the MediaPipe hand landmarker model if it doesn't already exist."""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading {MODEL_FILE_NAME}...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print(f"Downloaded {MODEL_FILE_NAME} to {MODEL_PATH}")
        except Exception as e:
            print(f"Error downloading model: {e}")
            print("Please ensure you have an internet connection or download the model manually.")
            exit()

def get_gesture(hand_landmarks):
    """
    Analyzes hand landmarks to detect a simple gesture.
    Detects a "Pointing Up" gesture if the index finger tip is notably higher than other finger tips.
    """
    if not hand_landmarks:
        return "No Hand"

    # Landmark indices for fingertips: Thumb(4), Index(8), Middle(12), Ring(16), Pinky(20)
    # Landmark indices for PIP joints (proximal interphalangeal): Thumb(2), Index(6), Middle(10), Ring(14), Pinky(18)
    
    # Get y-coordinates of fingertips (normalized to 0-1)
    thumb_tip_y = hand_landmarks[4].y
    index_tip_y = hand_landmarks[8].y
    middle_tip_y = hand_landmarks[12].y
    ring_tip_y = hand_landmarks[16].y
    pinky_tip_y = hand_landmarks[20].y

    # Get y-coordinates of PIP joints
    thumb_pip_y = hand_landmarks[2].y
    index_pip_y = hand_landmarks[6].y
    middle_pip_y = hand_landmarks[10].y
    ring_pip_y = hand_landmarks[14].y
    pinky_pip_y = hand_landmarks[18].y
    
    # Simple logic for "Pointing Up"
    # Check if index finger is extended (tip below its PIP joint, lower Y value means higher on screen)
    # And other fingers are relatively bent (tip above/close to its PIP joint, higher Y value means lower on screen)
    index_extended = index_tip_y < index_pip_y
    
    # Allow some tolerance for "bent" fingers, but ensure they are not extended
    flex_threshold = 0.03 # A small Y-difference for what constitutes "bent"

    thumb_bent = thumb_tip_y > thumb_pip_y - flex_threshold
    middle_bent = middle_tip_y > middle_pip_y - flex_threshold
    ring_bent = ring_tip_y > ring_pip_y - flex_threshold
    pinky_bent = pinky_tip_y > pinky_pip_y - flex_threshold

    if index_extended and thumb_bent and middle_bent and ring_bent and pinky_bent:
        return "Pointing Up!"
    
    # Simple check for open palm (all fingers extended)
    all_extended = (index_tip_y < index_pip_y) and \
                   (middle_tip_y < middle_pip_y) and \
                   (ring_tip_y < ring_pip_y) and \
                   (pinky_tip_y < pinky_pip_y)
    
    # A bit more lenient for thumb for "Open Palm"
    thumb_open = thumb_tip_y < thumb_pip_y + flex_threshold # Thumb can be slightly bent or straight

    if all_extended and thumb_open:
        return "Open Palm"

    return "No Specific Gesture"

def main():
    # Download the hand landmarker model if it doesn't exist
    download_model_if_not_exists()

    # Initialize MediaPipe Hand Landmarker
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create a HandLandmarker object.
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2 # Detect up to 2 hands
    )
    landmarker = HandLandmarker.create_from_options(options)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Flip the frame horizontally for a mirror effect, which is common for webcam apps
        frame = cv2.flip(frame, 1)
        
        # Get frame dimensions for drawing
        height, width, _ = frame.shape

        # Convert the BGR image to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create a MediaPipe Image object from the numpy array
        mp_image = mp.Image(image_format=mp.ImageFormat.RGB, data=rgb_frame)

        # Perform hand landmark detection
        hand_landmarker_result = landmarker.detect(mp_image)

        # Process the detection results
        if hand_landmarker_result.hand_landmarks:
            for hand_id, hand_landmarks_list in enumerate(hand_landmarker_result.hand_landmarks):
                # Draw landmarks
                for landmark in hand_landmarks_list:
                    x, y = int(landmark.x * width), int(landmark.y * height)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1) # Green circles for landmarks

                # Draw connections
                for connection in _HAND_CONNECTIONS:
                    start_node = hand_landmarks_list[connection[0]]
                    end_node = hand_landmarks_list[connection[1]]
                    start_point = (int(start_node.x * width), int(start_node.y * height))
                    end_point = (int(end_node.x * width), int(end_node.y * height))
                    cv2.line(frame, start_point, end_point, (255, 0, 0), 2) # Blue lines for connections
                
                # Get gesture for the first detected hand
                if hand_id == 0: # Only process gesture for the first hand for simplicity
                    gesture = get_gesture(hand_landmarks_list)
                    cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No Hand Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow("Gesture AI", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close() # Close the MediaPipe landmarker

if __name__ == "__main__":
    main()