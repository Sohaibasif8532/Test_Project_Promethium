import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

# Define the model path and URL for the hand landmarker model
MODEL_PATH = 'hand_landmarker.task'
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'

# Define Hand Landmark Connections manually for drawing using OpenCV.
# This is a list of tuples, where each tuple represents a connection between two landmark indices.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index finger
    (9, 10), (10, 11), (11, 12),              # Middle finger
    (13, 14), (14, 15), (15, 16),             # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20),    # Pinky finger
    (5, 9), (9, 13), (13, 17)                 # Palm base connections
]

# Landmark indices for clarity and readability
# THUMB
THUMB_MCP = 2  # Metacarpophalangeal joint
THUMB_TIP = 4  # Tip of the thumb

# INDEX FINGER
INDEX_MCP = 5
INDEX_PIP = 6  # Proximal interphalangeal joint
INDEX_TIP = 8

# MIDDLE FINGER
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_TIP = 12

# RING FINGER
RING_MCP = 13
RING_PIP = 14
RING_TIP = 16

# PINKY FINGER
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_TIP = 20

def download_model_if_not_exists(model_path, model_url):
    """
    Downloads the MediaPipe hand landmarker model if it doesn't already exist locally.
    """
    if not os.path.exists(model_path):
        print(f"Downloading {os.path.basename(model_path)}...")
        try:
            urllib.request.urlretrieve(model_url, model_path)
            print(f"Downloaded {os.path.basename(model_path)} to {model_path}")
        except Exception as e:
            print(f"Error downloading model: {e}")
            print("Please check your internet connection or the model URL.")
            exit()

def draw_landmarks_and_connections(image, hand_landmarks_list, connections):
    """
    Draws landmarks and their connections on the input image using OpenCV.

    Args:
        image: The OpenCV image (numpy array) to draw on.
        hand_landmarks_list: A list of lists of NormalizedLandmark objects,
                             where each inner list corresponds to one detected hand.
        connections: A list of tuples defining which landmarks to connect.
    Returns:
        The image with drawn landmarks and connections.
    """
    if not hand_landmarks_list:
        return image

    for landmarks_for_one_hand in hand_landmarks_list:
        # Draw connections
        for connection in connections:
            start_point = (int(landmarks_for_one_hand[connection[0]].x * image.shape[1]),
                           int(landmarks_for_one_hand[connection[0]].y * image.shape[0]))
            end_point = (int(landmarks_for_one_hand[connection[1]].x * image.shape[1]),
                         int(landmarks_for_one_hand[connection[1]].y * image.shape[0]))
            cv2.line(image, start_point, end_point, (0, 255, 0), 2) # Green color for connections

        # Draw landmarks (circles)
        for i, lm in enumerate(landmarks_for_one_hand):
            x, y = int(lm.x * image.shape[1]), int(lm.y * image.shape[0])
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1) # Red color for landmarks
    return image

def recognize_gesture(hand_landmarks_list, handedness_list):
    """
    Recognizes simple hand gestures based on landmark positions.

    Args:
        hand_landmarks_list: A list of lists of NormalizedLandmark objects for all detected hands.
        handedness_list: A list of lists of Category objects indicating 'Left' or 'Right' hand.
    Returns:
        A string describing the detected gesture(s) or "No Hand".
    """
    if not hand_landmarks_list:
        return "No Hand"

    gestures = []
    for i, landmarks in enumerate(hand_landmarks_list):
        current_handedness = handedness_list[i][0].category_name # e.g., 'Left' or 'Right'

        # Fetch landmarks using defined constants for clarity
        thumb_tip = landmarks[THUMB_TIP]
        thumb_mcp = landmarks[THUMB_MCP]
        index_tip = landmarks[INDEX_TIP]
        index_pip = landmarks[INDEX_PIP]
        middle_tip = landmarks[MIDDLE_TIP]
        middle_pip = landmarks[MIDDLE_PIP]
        ring_tip = landmarks[RING_TIP]
        ring_pip = landmarks[RING_PIP]
        pinky_tip = landmarks[PINKY_TIP]
        pinky_pip = landmarks[PINKY_PIP]

        # Y-coordinate threshold for finger extension.
        # A smaller Y-value means higher on the image.
        Y_THRESHOLD = 0.05

        # Check if a finger is "extended" by comparing its tip's Y-coordinate
        # to its PIP joint's Y-coordinate.
        is_index_extended = index_tip.y < index_pip.y - Y_THRESHOLD
        is_middle_extended = middle_tip.y < middle_pip.y - Y_THRESHOLD
        is_ring_extended = ring_tip.y < ring_pip.y - Y_THRESHOLD
        is_pinky_extended = pinky_tip.y < pinky_pip.y - Y_THRESHOLD

        # Thumb extension is more complex due to its unique articulation and rotation.
        # This checks if the thumb tip is significantly higher than its MCP joint
        # AND extends horizontally away from the palm, based on handedness.
        is_thumb_extended = False
        if current_handedness == 'Left':
             # For a left hand, the thumb usually extends to the left (smaller X) and upwards (smaller Y)
             is_thumb_extended = (thumb_tip.x < thumb_mcp.x - 0.03) and (thumb_tip.y < thumb_mcp.y)
        elif current_handedness == 'Right':
             # For a right hand, the thumb usually extends to the right (larger X) and upwards (smaller Y)
             is_thumb_extended = (thumb_tip.x > thumb_mcp.x + 0.03) and (thumb_tip.y < thumb_mcp.y)

        gesture = "Unknown"

        # Gesture Logic:
        if is_index_extended and is_middle_extended and not is_ring_extended and not is_pinky_extended and not is_thumb_extended:
            gesture = "Peace Sign"
        elif is_thumb_extended and not is_index_extended and not is_middle_extended and not is_ring_extended and not is_pinky_extended:
            gesture = "Thumbs Up"
        elif is_index_extended and is_middle_extended and is_ring_extended and is_pinky_extended and is_thumb_extended:
            gesture = "Open Hand"
        elif not is_index_extended and not is_middle_extended and not is_ring_extended and not is_pinky_extended and not is_thumb_extended:
             # If all fingers (and thumb) are not extended, it's likely a closed fist.
            gesture = "Closed Fist"
        elif is_index_extended and not is_middle_extended and not is_ring_extended and not is_pinky_extended and not is_thumb_extended:
            gesture = "Pointing Index"
        # Additional gestures can be added here with more specific landmark checks.

        gestures.append(f"{current_handedness} Hand: {gesture}")
    
    # Return a comma-separated string of all detected gestures
    return ", ".join(gestures) if gestures else "No Hand Detected"

def main():
    """
    Main function to initialize the camera, MediaPipe Hand Landmarker,
    process frames, recognize gestures, and display results.
    """
    download_model_if_not_exists(MODEL_PATH, MODEL_URL)

    # Initialize MediaPipe Hand Landmarker
    # Configure to detect up to 2 hands.
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    # Initialize OpenCV camera capture
    cap = cv2.VideoCapture(0) # 0 for the default camera

    if not cap.isOpened():
        print("Error: Could not open video stream. Make sure camera is connected and not in use.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Flip the frame horizontally for a more intuitive mirror-like view
        frame = cv2.flip(frame, 1)

        # Convert the BGR image (from OpenCV) to RGB (as MediaPipe expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Create a MediaPipe Image object from the RGB frame
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Perform hand landmark detection
        detection_result = detector.detect(mp_image)

        gesture_text = "No Hand Detected"

        if detection_result.hand_landmarks:
            all_hand_landmarks = detection_result.hand_landmarks
            all_handedness = detection_result.handedness

            # Draw landmarks and connections for all detected hands
            frame = draw_landmarks_and_connections(frame, all_hand_landmarks, HAND_CONNECTIONS)

            # Recognize gesture for all detected hands
            gesture_text = recognize_gesture(all_hand_landmarks, all_handedness)
        
        # Display the recognized gesture text on the frame
        cv2.putText(frame, f"Gesture: {gesture_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow('Gesture Recognition AI', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()