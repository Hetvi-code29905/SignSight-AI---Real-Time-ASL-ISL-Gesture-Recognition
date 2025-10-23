# SignSight AI

SignSight AI is a real-time system designed to interpret and translate static American Sign Language (ASL) gestures using a standard computer webcam. Leveraging modern computer vision and machine learning techniques, it recognizes one- and two-handed gestures and provides immediate visual feedback, including the predicted sign and a confidence score.

---

## 📖 Overview

Visual languages like ASL rely on precise handshapes, movements, and positions. Recognizing these gestures accurately and in real-time presents a significant computer vision challenge. SignSight AI addresses this by providing an accessible tool that uses standard webcams and sophisticated software to bridge communication gaps.

The system transforms webcam input into meaningful gesture recognition through the following key tasks:

* **Robust Hand Detection:** Reliably locates and isolates hands within the video frame, even with cluttered backgrounds or varying lighting.
* **Detailed Pose Analysis:** Extracts precise skeletal landmark data from detected hands to capture finger positioning and hand orientation nuances.
* **Accurate Gesture Classification:** Identifies specific ASL signs using the extracted landmark features.
* **Intuitive Visualization:** Displays the detected hand skeletons, a bounding box highlighting the active signing area, and the recognized sign with its confidence level.

---

## ⚙️ How It Works

SignSight AI uses a streamlined, multi-stage pipeline optimized for real-time processing:

### 1. Hand Tracking & Landmark Extraction (MediaPipe)

* Processes the webcam feed frame by frame.
* MediaPipe Hands detects up to two hands and extracts 21 precise 3D landmarks per hand (x, y coordinates used in this version).

### 2. Feature Processing & Normalization (NumPy)

* Landmark coordinates are translated relative to the wrist (landmark 0) and scaled based on maximum distance from the wrist.
* Each hand provides 42 normalized features, combined into an 84-feature vector.
* If only one hand is present, the second 42 features are padded with zeros to maintain a fixed input size.

### 3. Gesture Classification (MLP)

* The 84-feature vector is fed into a pre-trained Multi-Layer Perceptron (MLP) implemented in TensorFlow/Keras.
* Outputs probabilities for each known gesture class.

### 4. Visualization (OpenCV)

* Displays the original webcam feed with:

  * Skeletal structure of detected hands.
  * Bounding box encompassing all detected hands.
  * Recognized ASL gesture with confidence percentage (e.g., `hello (98.7%)`).

---

## 📊 Training Data & Augmentation

### Dataset

* Derived from a custom image dataset representing everyday words and phrases.
* Approximately 35-40 images per class across 50 ASL gesture classes.
* Original dataset credit: Saipatwar et al., 2025 (modified and expanded for this project).

### Image Augmentation (Albumentations)

* Applied to enhance generalization and prevent overfitting:

  * Rotation (±20°)
  * RandomBrightnessContrast
  * GaussNoise
  * ShiftScaleRotate
  * Perspective warping
  * HorizontalFlip
* Landmarks were extracted from both original and augmented images to form `hand_gesture_data_84_img_aug.pickle`.

---

## 🧠 Model Architecture & Results

* **Model:** Feed-forward MLP using TensorFlow/Keras Sequential API
* **Input:** 84-feature vector
* **Hidden Layers:** Dense layers (e.g., 256 units, 128 units) with ReLU and Dropout (0.5)
* **Output Layer:** Dense with `num_classes` units, softmax activation
* **Training:** Adam optimizer, categorical_crossentropy loss, EarlyStopping, ReduceLROnPlateau
* **Performance:** 92.62% accuracy on held-out test set

**Saved Files:**

* Trained Model: `gesture_model_84_img_aug.h5`
* Label Encoder: `mlp_label_encoder_84_img_aug.pickle`

---

## 💻 Tech Stack

* **Python:** Version 3.8+ (64-bit recommended)
* **TensorFlow / Keras:** Core deep learning framework for the MLP model
* **MediaPipe:** High-performance hand tracking and landmark extraction
* **OpenCV (cv2):** Image reading/writing, color conversions, drawing, and local webcam interaction
* **NumPy:** Numerical computations and array manipulation
* **Scikit-learn:** LabelEncoder and train_test_split
* **Albumentations:** Image augmentation library
* **Pickle:** Serializing and loading Python objects

---


## 🔭 Future Scope & Enhancements

* **Dynamic Gesture Recognition:** Use RNNs or Transformers for motion-based gestures.
* **Alternative Hand Detection:** Fine-tune YOLO or other detectors.
* **Vocabulary Expansion:** Include more ASL signs.
* **Web Deployment (TensorFlow.js):** Run entirely in-browser.
* **3D Landmarks:** Incorporate Z-coordinate for robustness.
* **User Feedback:** Allow corrections to improve the model.

---

*SignSight AI bridges communication gaps by making ASL recognition accessible, accurate, and real-time.*
