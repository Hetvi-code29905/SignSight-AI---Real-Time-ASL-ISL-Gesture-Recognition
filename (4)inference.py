import tensorflow as tf
import mediapipe as mp
import cv2 
import numpy as np
import pickle
import time
import os



#Define paths relative to the script location
script_dir = os.path.dirname(os.path.abspath(__file__)) # Get directory of the script
model_path = os.path.join(script_dir, 'gesture_model_84_img_aug.h5')
encoder_path = os.path.join(script_dir, 'mlp_label_encoder_84_img_aug.pickle')

# Load the model
try:
    gesture_classifier = tf.keras.models.load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Ensure '{os.path.basename(model_path)}' is in the same directory as the script.")
    exit()

# Load the label encoder
try:
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
except Exception as e:
    print(f"Error loading label encoder: {e}")
    print(f"Ensure '{os.path.basename(encoder_path)}' is in the same directory as the script.")
    exit()

print("Models and encoder loaded successfully.")


# MediaPipe Hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(
    static_image_mode=False,       
    max_num_hands=2,               
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5   
)

#Helper function for normalization
def normalize_landmarks(landmarks_list):
    """ Normalizes a flat list of 42 landmarks (from MediaPipe). """
    landmarks_np = np.array(landmarks_list).reshape(-1, 2)
    if landmarks_np.size == 0: return np.zeros(42)
    wrist_origin = landmarks_np[0].copy()
    landmarks_translated = landmarks_np - wrist_origin
    max_val = np.max(np.abs(landmarks_translated))
    if max_val == 0: max_val = 1
    landmarks_normalized = landmarks_translated / max_val
    return landmarks_normalized.flatten()

#Start Webcam using OpenCV
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started. Press 'q' to quit.")

#Main loop: MediaPipe (Detect + Landmarks) -> MLP (Classify)
try:
    while True:
        # Read frame from webcam
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the frame horizontally for a later selfie-view display
        # and convert the BGR image to RGB.
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame_rgb.flags.writeable = False
        results = hands_detector.process(frame_rgb)
        frame_rgb.flags.writeable = True # Image is now writeable again

        # Prepare for prediction
        final_landmark_vector = np.zeros(84) 
        hand_count = 0
        all_hand_boxes = []

        # Draw the hand annotations on the image.
        frame.flags.writeable = True

        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)

            # Extract, Normalize, and Draw Hand 1
            hand_landmarks_1 = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                frame, hand_landmarks_1, mp_hands.HAND_CONNECTIONS)

            hand1_raw = []
            x_coords_1, y_coords_1 = [], []
            h, w, _ = frame.shape 
            for lm in hand_landmarks_1.landmark:
                hand1_raw.extend([lm.x, lm.y])
                x_coords_1.append(lm.x)
                y_coords_1.append(lm.y)

            # Ensure landmarks were extracted before normalizing
            if hand1_raw:
                final_landmark_vector[0:42] = normalize_landmarks(hand1_raw)
                hand_count = 1
                # Calculate bounding box (ensure coords are valid)
                if x_coords_1 and y_coords_1:
                     all_hand_boxes.append((
                         int(min(x_coords_1) * w) - 20, int(min(y_coords_1) * h) - 20,
                         int(max(x_coords_1) * w) + 20, int(max(y_coords_1) * h) + 20
                     ))

            if num_hands == 2:
                #Extract, Normalize, and Draw Hand 2
                hand_landmarks_2 = results.multi_hand_landmarks[1]
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks_2, mp_hands.HAND_CONNECTIONS)

                hand2_raw = []
                x_coords_2, y_coords_2 = [], []
                for lm in hand_landmarks_2.landmark:
                    hand2_raw.extend([lm.x, lm.y])
                    x_coords_2.append(lm.x)
                    y_coords_2.append(lm.y)

                # Ensure landmarks were extracted before normalizing
                if hand2_raw:
                    final_landmark_vector[42:84] = normalize_landmarks(hand2_raw)
                    hand_count = 2
                    # Calculate bounding box (ensure coords are valid)
                    if x_coords_2 and y_coords_2:
                        all_hand_boxes.append((
                            int(min(x_coords_2) * w) - 20, int(min(y_coords_2) * h) - 20,
                            int(max(x_coords_2) * w) + 20, int(max(y_coords_2) * h) + 20
                        ))

        #STAGE 2: Gesture Classification (One prediction for the whole frame)
        phrase = "" 
        if hand_count > 0 and all_hand_boxes: 
            input_data = np.array([final_landmark_vector], dtype='float32')

            prediction = gesture_classifier.predict(input_data, verbose=0)
            predicted_index = np.argmax(prediction)
            # Check if index is valid for the loaded encoder
            if predicted_index < len(label_encoder.classes_):
                 phrase = label_encoder.inverse_transform([predicted_index])[0]
                 confidence = np.max(prediction)
                 display_text = f'{phrase} ({confidence*100:.1f}%)'
            else:
                 display_text = "Unknown Index"


            #Drawing Logic
            if hand_count == 1:
                (x1, y1, x2, y2) = all_hand_boxes[0]
            elif hand_count == 2 and len(all_hand_boxes) == 2: # Check if two boxes were actually created
                # Combine the two boxes
                (x1_1, y1_1, x2_1, y2_1) = all_hand_boxes[0]
                (x1_2, y1_2, x2_2, y2_2) = all_hand_boxes[1]
                x1 = min(x1_1, x1_2)
                y1 = min(y1_1, y1_2)
                x2 = max(x2_1, x2_2)
                y2 = max(y2_1, y2_2)
            else: # Fallback if boxes mismatch count
                x1, y1, x2, y2 = 50, 50, 150, 150 # Default box? or handle error

            # Ensure coordinates are within frame boundaries before drawing
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)


            # Draw the single prediction and the combined box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, display_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Hand Gesture Recognition', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred during the main loop: {e}")

finally:
    # Release the webcam and destroy all windows
    cap.release()
    cv2.destroyAllWindows()
    print("\nWebcam stopped and windows closed.")