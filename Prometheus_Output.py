import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
# static_image_mode=False for video stream
# max_num_hands=1 to detect only one hand for simplicity
# min_detection_confidence and min_tracking_confidence set to reasonable values
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Gesture recognition logic
def recognize_gesture(hand_landmarks, handedness_label):
    """
    Recognizes a hand gesture based on MediaPipe hand landmarks and handedness.

    Args:
        hand_landmarks: A MediaPipe HandLandmarks object containing 21 hand landmarks.
        handedness_label: A string indicating "Left" or "Right" hand.

    Returns:
        A string representing the recognized gesture (e.g., "Open Hand", "Fist", "Thumb Up").
    """
    if not hand_landmarks:
        return "No Hand"

    landmarks = hand_landmarks.landmark
    
    # Store boolean for each finger: [Thumb, Index, Middle, Ring, Pinky]
    finger_is_extended = [False] * 5 

    # --- THUMB (index 0 in finger_is_extended) ---
    # Thumb extension is checked by combining vertical (Y-axis) and horizontal (X-axis) position.
    # Vertical check: THUMB_TIP (4) Y-coord is higher (smaller value) than THUMB_IP (3) and THUMB_MCP (2).
    # Horizontal check: THUMB_TIP (4) X-coord is 'outward' from the palm, relative to THUMB_MCP (2).
    
    is_thumb_vertically_extended = landmarks[mp_hands.HandLandmark.THUMB_TIP].y < landmarks[mp_hands.HandLandmark.THUMB_IP].y and \
                                   landmarks[mp_hands.HandLandmark.THUMB_IP].y < landmarks[mp_hands.HandLandmark.THUMB_MCP].y

    is_thumb_horizontally_extended = False
    if handedness_label == "Right":
        # For a right hand, an extended thumb's tip (4) will have a smaller X-coordinate than its MCP (2)
        # (i.e., it's to the left when viewed frontally by the camera)
        is_thumb_horizontally_extended = landmarks[mp_hands.HandLandmark.THUMB_TIP].x < landmarks[mp_hands.HandLandmark.THUMB_MCP].x
    else: # Left hand
        # For a left hand, an extended thumb's tip (4) will have a larger X-coordinate than its MCP (2)
        # (i.e., it's to the right when viewed frontally by the camera)
        is_thumb_horizontally_extended = landmarks[mp_hands.HandLandmark.THUMB_TIP].x > landmarks[mp_hands.HandLandmark.THUMB_MCP].x

    if is_thumb_vertically_extended and is_thumb_horizontally_extended:
        finger_is_extended[0] = True


    # --- Other fingers (Index, Middle, Ring, Pinky - indices 1-4 in finger_is_extended) ---
    # A finger is considered extended if its tip (Y-coordinate) is significantly higher 
    # (smaller Y-value) than its PIP joint, and its PIP joint is higher than its MCP joint.
    # This heuristic checks for a relatively straight finger pointing upwards/outwards.
    
    # Index finger (1)
    if landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and \
       landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y < landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y:
        finger_is_extended[1] = True
    
    # Middle finger (2)
    if landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and \
       landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y:
        finger_is_extended[2] = True
    
    # Ring finger (3)
    if landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y and \
       landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y < landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].y:
        finger_is_extended[3] = True
    
    # Pinky finger (4)
    if landmarks[mp_hands.HandLandmark.PINKY_TIP].y < landmarks[mp_hands.HandLandmark.PINKY_PIP].y and \
       landmarks[mp_hands.HandLandmark.PINKY_PIP].y < landmarks[mp_hands.HandLandmark.PINKY_MCP].y:
        finger_is_extended[4] = True

    # --- Gesture Classification based on the `finger_is_extended` list ---
    # The order of these checks matters: more specific gestures should be checked before more general ones.
    # finger_is_extended format: [Thumb, Index, Middle, Ring, Pinky]

    # I Love You (Thumb, Index, Pinky extended; Middle, Ring curled)
    if finger_is_extended[0] and finger_is_extended[1] and not finger_is_extended[2] and not finger_is_extended[3] and finger_is_extended[4]:
        return "I Love You"
    # Rock On (Index, Pinky extended; Thumb, Middle, Ring curled)
    elif not finger_is_extended[0] and finger_is_extended[1] and not finger_is_extended[2] and not finger_is_extended[3] and finger_is_extended[4]:
        return "Rock On"
    # Thumb Up (Only Thumb extended)
    elif finger_is_extended[0] and not finger_is_extended[1] and not finger_is_extended[2] and not finger_is_extended[3] and not finger_is_extended[4]:
        return "Thumb Up"
    # Peace Sign (Index, Middle extended; Thumb, Ring, Pinky curled)
    elif not finger_is_extended[0] and finger_is_extended[1] and finger_is_extended[2] and not finger_is_extended[3] and not finger_is_extended[4]:
        return "Peace Sign"
    # Pointing (Only Index extended)
    elif not finger_is_extended[0] and finger_is_extended[1] and not finger_is_extended[2] and not finger_is_extended[3] and not finger_is_extended[4]:
        return "Pointing"
    # Open Hand (All fingers extended)
    elif all(finger_is_extended):
        return "Open Hand"
    # Fist (No fingers extended)
    elif not any(finger_is_extended):
        return "Fist"

    # Default if no specific gesture is recognized
    return "Unknown Gesture"

# --- Main program loop to capture video and perform gesture recognition ---
cap = cv2.VideoCapture(0) # 0 for default webcam

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Flip the frame horizontally for a selfie-view display, which is more intuitive
    frame = cv2.flip(frame, 1)

    # Convert the BGR image from OpenCV to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands to detect hand landmarks
    results = hands.process(rgb_frame)

    gesture = "No Hand Detected"

    # If hands are detected, draw landmarks and recognize the gesture
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Draw the hand landmarks and connections on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get handedness (e.g., "Left" or "Right") which is crucial for thumb logic
            handedness_label = results.multi_handedness[idx].classification[0].label
            
            # Recognize the gesture using the defined logic
            gesture = recognize_gesture(hand_landmarks, handedness_label)

    # Display the recognized gesture on the top-left corner of the frame
    cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Gesture AI', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()