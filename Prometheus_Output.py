import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import urllib.request
import numpy as np
import time

# --- Global variables for detection results ---
detection_results = None
frame_timestamp_ms = 0

# --- Callback function for live stream mode ---
def results_callback(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global detection_results, frame_timestamp_ms
    detection_results = result
    frame_timestamp_ms = timestamp_ms

# --- Model Download and Setup ---
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
MODEL_PATH = 'hand_landmarker.task'

def download_model(model_url, model_path):
    if not os.path.exists(model_path):
        print(f"Downloading {os.path.basename(model_url)}...")
        urllib.request.urlretrieve(model_url, model_path)
        print("Download complete.")
    else:
        print(f"{os.path.basename(model_url)} already exists.")

download_model(MODEL_URL, MODEL_PATH)

# --- MediaPipe Hand Landmarker Setup ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=results_callback
)
detector = HandLandmarker.create_from_options(options)

# --- Drawing Utilities ---
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17), (0, 17), (0, 5) # Palm base connections
]

def draw_landmarks_on_image(rgb_image, hand_landmarks_list, hand_connections):
    annotated_image = np.copy(rgb_image)
    image_h, image_w, _ = annotated_image.shape

    if hand_landmarks_list:
        for hand_landmarks in hand_landmarks_list:
            # Draw connections
            for connection in hand_connections:
                start_node = hand_landmarks[connection[0]]
                end_node = hand_landmarks[connection[1]]
                start_point = (int(start_node.x * image_w), int(start_node.y * image_h))
                end_point = (int(end_node.x * image_w), int(end_node.y * image_h))
                cv2.line(annotated_image, start_point, end_point, (0, 255, 0), 2)

            # Draw landmarks
            for landmark in hand_landmarks:
                center = (int(landmark.x * image_w), int(landmark.y * image_h))
                cv2.circle(annotated_image, center, 5, (0, 0, 255), -1) # Red filled circle

    return annotated_image

# --- Gesture Recognition Logic ---
FINGER_TIPS = {
    "Thumb": 4,
    "Index": 8,
    "Middle": 12,
    "Ring": 16,
    "Pinky": 20
}

FINGER_MCPS = { # Metacarpophalangeal joints (base of fingers)
    "Thumb": 2, # Thumb's MCP is landmark 2 (Proximal Phalanx base)
    "Index": 5,
    "Middle": 9,
    "Ring": 13,
    "Pinky": 17
}

# Heuristic to check if a finger is extended (up)
# This assumes an upright hand and camera Y-axis pointing down.
# A finger is up if its tip's Y coordinate is significantly less (higher on screen)
# than its MCP joint's Y coordinate.
def is_finger_extended(landmarks, finger_name):
    tip_idx = FINGER_TIPS[finger_name]
    mcp_idx = FINGER_MCPS[finger_name]
    
    # Using a small offset for clearer distinction, adjust as needed
    # This offset accounts for slight bends or detection noise.
    y_offset = 0.02 

    # Compare the Y-coordinate of the tip with its MCP joint.
    # If the tip is 'above' (smaller Y value) the MCP, it's considered extended.
    return landmarks[tip_idx].y < landmarks[mcp_idx].y - y_offset


def recognize_gesture(hand_landmarks_list, handedness_list):
    gestures = []
    if not hand_landmarks_list:
        return gestures

    for i, hand_landmarks in enumerate(hand_landmarks_list):
        handedness = handedness_list[i][0].category_name

        fingers_up_status = {}
        for finger_name in FINGER_TIPS.keys():
            fingers_up_status[finger_name] = is_finger_extended(hand_landmarks, finger_name)
        
        num_fingers_up = sum(fingers_up_status.values())

        current_gesture = "Unknown"
        
        if num_fingers_up == 0:
            current_gesture = "Fist"
        elif num_fingers_up == 1:
            if fingers_up_status["Index"]:
                current_gesture = "Pointing"
            elif fingers_up_status["Thumb"]:
                current_gesture = "Thumbs Up"
        elif num_fingers_up == 2:
            if fingers_up_status["Index"] and fingers_up_status["Middle"]:
                current_gesture = "Peace/Victory"
            elif fingers_up_status["Thumb"] and fingers_up_status["Pinky"]:
                current_gesture = "Shaka"
        elif num_fingers_up == 3:
             current_gesture = "Three Fingers Up"
        elif num_fingers_up == 4:
            current_gesture = "Four Fingers Up"
        elif num_fingers_up == 5:
            current_gesture = "Open Hand"
        
        gestures.append(f"{handedness}: {current_gesture}")

    return gestures

# --- Main Webcam Loop ---
def main():
    global detection_results, frame_timestamp_ms

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam opened successfully. Press 'q' to quit.")

    # Variables for FPS calculation
    prev_frame_time = 0
    new_frame_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Flip frame horizontally for a "selfie-view"
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB as required by MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Get current timestamp for MediaPipe (milliseconds)
        current_frame_timestamp = int(time.time() * 1000)

        # Send image to MediaPipe for async detection
        detector.detect_async(mp_image, current_frame_timestamp)

        # Initialize annotated_frame with original frame
        annotated_frame = frame.copy()

        # Process results from the callback if available and matching current timestamp
        if detection_results and frame_timestamp_ms == current_frame_timestamp:
            if detection_results.hand_landmarks:
                annotated_frame = draw_landmarks_on_image(
                    annotated_frame, detection_results.hand_landmarks, HAND_CONNECTIONS
                )

                # Recognize gesture
                gestures = recognize_gesture(detection_results.hand_landmarks, detection_results.handedness)
                for i, gesture_text in enumerate(gestures):
                    cv2.putText(annotated_frame, gesture_text, (10, 30 + i * 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(annotated_frame, "No hand detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            # If no fresh results, show a message or just the raw frame
            cv2.putText(annotated_frame, "Detecting...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        # Calculate and display FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (frame.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Gesture AI', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close() # Close the detector when done

if __name__ == '__main__':
    main()