import pickle
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

#Load the landmark data derived from augmented images
print("Loading the image-augmented 84-feature landmark data...")
with open('your_instance/hand_gesture_data_84_img_aug.pickle', 'rb') as f:
    dataset = pickle.load(f)

data = np.array(dataset['data'])
labels = np.array(dataset['labels'])
print(f"Loaded {len(data)} total landmark samples.")

#Filter classes with too few samples
min_samples_per_class = 5
class_counts = Counter(labels)
filtered_data = []
filtered_labels = []
for i in range(len(data)):
    label = labels[i]
    if class_counts[label] >= min_samples_per_class:
        filtered_data.append(data[i])
        filtered_labels.append(label)
data = np.array(filtered_data)
labels = np.array(filtered_labels)
print(f"Proceeding with {len(data)} samples from {len(np.unique(labels))} classes.")

#Prepare data
X = data
y = labels

#Encode labels and split the data
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.2,
    random_state=42,
    stratify=y_categorical
)

#Save the correct encoder file for this dataset
with open('your_instance/mlp_label_encoder_84_img_aug.pickle', 'wb') as f:
    pickle.dump(label_encoder, f)
print("MLP LabelEncoder (84-feature, image augmented) saved.")

print("\nData is now correctly prepared for the MLP Classifier:")
#Corrected shape comment
print(f"X_train shape: {X_train.shape}") 
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")