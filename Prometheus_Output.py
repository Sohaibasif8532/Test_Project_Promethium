import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
# max_num_hands: Detects up to 1 hand for simplicity, as the task implies a single user interaction.
# min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for hand detection to be considered successful.
# min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) for the hand landmarks to be tracked successfully.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
# Initialize MediaPipe Drawing utilities for visualizing landmarks
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define landmark indices for clarity (based on MediaPipe's hand landmark model)
# Thumb
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

# Index finger
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

# Middle finger
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12

# Ring finger
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16

# Pinky finger
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

# Function to determine if a finger (index, middle, ring, pinky) is extended.
# This heuristic checks if the tip of the finger is significantly higher (smaller Y-coordinate)
# than its PIP (Proximal Interphalangeal) and MCP (Metacarpophalangeal) joints.
# This works well when the hand is generally oriented upright, with fingers pointing upwards.
def is_finger_extended(landmark_list, tip_idx, pip_idx, mcp_idx):
    return landmark_list[tip_idx].y < landmark_list[pip_idx].y and \
           landmark_list[tip_idx].y < landmark_list[mcp_idx].y

# Function to determine if the thumb is extended.
# The thumb's movement is unique. This heuristic primarily checks if the thumb tip's Y-coordinate
# is higher than its IP (Interphalangeal) and MCP joints. This helps detect "thumbs up" or open hand.
def is_thumb_extended(landmark_list):
    return landmark_list[THUMB_TIP].y < landmark_list[THUMB_IP].y and \
           landmark_list[THUMB_TIP].y < landmark_list[THUMB_MCP].y

# Main loop for webcam feed processing
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a natural, mirror-like view.
    frame = cv2.flip(frame, 1)

    # Convert the BGR image (OpenCV default) to RGB (MediaPipe required).
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands to detect hand landmarks.
    results = hands.process(rgb_frame)

    # Initialize gesture text.
    gesture = "No Hand Detected"

    # Check if hand landmarks were detected.
    if results.multi_hand_landmarks:
        # Loop through each detected hand (max_num_hands is set to 1, so this loop runs once).
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the detected hand landmarks and connections on the frame.
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the list of landmark objects.
            lm_list = hand_landmarks.landmark

            # Determine the extension state for each finger.
            thumb_ext = is_thumb_extended(lm_list)
            index_ext = is_finger_extended(lm_list, INDEX_TIP, INDEX_PIP, INDEX_MCP)
            middle_ext = is_finger_extended(lm_list, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP)
            ring_ext = is_finger_extended(lm_list, RING_TIP, RING_PIP, RING_MCP)
            pinky_ext = is_finger_extended(lm_list, PINKY_TIP, PINKY_PIP, PINKY_MCP)

            # Store finger states as a list (True for extended, False for bent).
            finger_states = [thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext]

            # --- Gesture Recognition Logic ---
            # These rules define various gestures based on finger extension states.
            
            # Open Hand (Paper gesture): All fingers are extended.
            if all(finger_states):
                gesture = "Open Hand (Paper)"
            # Closed Fist (Rock gesture): All fingers (including thumb) are bent/tucked.
            elif not any(finger_states): # If no finger is extended
                gesture = "Closed Fist (Rock)"
            # Peace Sign (V-sign): Index and Middle fingers extended, others bent.
            elif index_ext and middle_ext and not ring_ext and not pinky_ext and not thumb_ext:
                gesture = "Peace Sign (V)"
            # Thumbs Up: Only the thumb is extended, others are bent.
            elif thumb_ext and not index_ext and not middle_ext and not ring_ext and not pinky_ext:
                gesture = "Thumbs Up"
            # Pointing Up: Only the index finger is extended, others are bent.
            elif index_ext and not middle_ext and not ring_ext and not pinky_ext and not thumb_ext:
                gesture = "Pointing Up"
            # If none of the defined gestures match.
            else:
                gesture = "Unknown Gesture"

    # Display the recognized gesture on the frame.
    cv2.putText(frame, "Gesture: " + gesture, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the processed frame.
    cv2.imshow("Gesture AI", frame)

    # Exit the loop if the 'q' key is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all OpenCV windows.
cap.release()
cv2.destroyAllWindows()