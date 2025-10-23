import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
import albumentations as A
import random # Import random module for selecting images


# Configuration
DATA_DIR = 'your_data_set_link_here'
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
NUM_IMAGES_TO_AUGMENT_PER_CLASS = 15 # How many images to randomly select for augmentation

#Define Augmentations
transform = A.Compose([
    A.Rotate(limit=20, p=0.6, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
    A.GaussNoise(var_limit=(10.0, 60.0), p=0.4),
    A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.Perspective(scale=(0.02, 0.06), p=0.3),
    A.HorizontalFlip(p=0.5),
])

#Data storage
data = []
labels = []
processed_counts = {}

print(f"Starting 84-feature extraction (All Originals + {NUM_IMAGES_TO_AUGMENT_PER_CLASS} Augmented per Class)...")

def process_and_extract_landmarks(image_rgb, label):
    """Processes an image (original or augmented) and returns landmarks/label if successful."""
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        all_hands_landmarks = []
        num_hands = len(results.multi_hand_landmarks)
        # Process hand 1
        hand1_landmarks = []
        for landmark in results.multi_hand_landmarks[0].landmark:
            hand1_landmarks.extend([landmark.x, landmark.y])
        norm_lm1 = normalize_landmarks(hand1_landmarks)
        all_hands_landmarks.extend(norm_lm1)
        # Process hand 2 or pad
        if num_hands == 2:
            hand2_landmarks = []
            for landmark in results.multi_hand_landmarks[1].landmark:
                hand2_landmarks.extend([landmark.x, landmark.y])
            norm_lm2 = normalize_landmarks(hand2_landmarks)
            all_hands_landmarks.extend(norm_lm2)
        else:
            all_hands_landmarks.extend(np.zeros(42))
        # Check success
        if np.any(all_hands_landmarks):
            return all_hands_landmarks, label
    return None, None 

def normalize_landmarks(landmarks_list):
    """ Normalizes a flat list of 42 landmarks. """
    landmarks_np = np.array(landmarks_list).reshape(-1, 2)
    if landmarks_np.size == 0: return np.zeros(42)
    wrist_origin = landmarks_np[0].copy()
    landmarks_translated = landmarks_np - wrist_origin
    max_val = np.max(np.abs(landmarks_translated))
    if max_val == 0: max_val = 1
    landmarks_normalized = landmarks_translated / max_val
    return landmarks_normalized.flatten()

#Main Loop
for dir_ in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(class_path): continue

    print(f'\nProcessing class: {dir_}')
    processed_counts[dir_] = {'original': 0, 'augmented': 0}
    # Get a list of all image file paths in the current class directory
    all_image_paths_in_class = [os.path.join(class_path, f) for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]

    #Step 1: Process ALL original images in the class
    print(f'  Processing {len(all_image_paths_in_class)} original images...')
    successful_original_paths = [] 
    for full_img_path in all_image_paths_in_class:
        img_path_basename = os.path.basename(full_img_path)
        try:
            img = cv2.imread(full_img_path)
            if img is None:
                print(f"  Warning: Could not read original image {img_path_basename}. Skipping.")
                continue
            img_rgb_original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Process the ORIGINAL image
            original_landmarks, original_label = process_and_extract_landmarks(img_rgb_original, dir_)
            if original_landmarks is not None:
                data.append(original_landmarks)
                labels.append(original_label)
                processed_counts[dir_]['original'] += 1
                successful_original_paths.append(full_img_path)

        except Exception as e:
             print(f"  Error processing original image {img_path_basename}: {e}")

    #Step 2: Select a subset of ORIGINAL paths and process augmented versions
    num_to_augment = min(NUM_IMAGES_TO_AUGMENT_PER_CLASS, len(successful_original_paths)) 
    if num_to_augment > 0:
        print(f'  Selecting {num_to_augment} images for augmentation...')
        # Randomly choose paths from the list of successful originals
        paths_to_augment = random.sample(successful_original_paths, num_to_augment)

        print(f'  Processing augmentations for {len(paths_to_augment)} images...')
        for full_img_path in paths_to_augment:
            img_path_basename = os.path.basename(full_img_path)
            try:
                img = cv2.imread(full_img_path)
                if img is None: continue
                img_rgb_original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Apply augmentation
                augmented = transform(image=img_rgb_original)
                img_rgb_augmented = augmented['image']

                # Process augmented image
                augmented_landmarks, augmented_label = process_and_extract_landmarks(img_rgb_augmented, dir_)
                if augmented_landmarks is not None:
                    data.append(augmented_landmarks)
                    labels.append(augmented_label)
                    processed_counts[dir_]['augmented'] += 1
                    # print(f'    ✓ Augmented: {img_path_basename}')

            except Exception as e:
                 print(f"  Error augmenting image {img_path_basename}: {e}")

#Final Summary and Save
print(f"\n----------------------------------")
print(f"Feature extraction complete. Found {len(data)} total samples (All Originals + Selected Augmented).")
print(f"----------------------------------")
print("Processed samples per class (Original / Augmented):")
for class_name, counts in processed_counts.items():
    print(f"- {class_name}: {counts['original']} / {counts['augmented']}")
print(f"----------------------------------")

# Save the combined dataset (using the same filename as before)
with open('your_instance/hand_gesture_data_84_img_aug.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Combined (All Originals + Selected Augmented) landmark data saved to hand_gesture_data_84_img_aug.pickle")