import pickle
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

#Load the landmark data derived from augmented images
print("Loading the image-augmented 84-feature landmark data...")
# Correctly loading the data generated in the previous step
with open('your_instance/hand_gesture_data_84_img_aug.pickle', 'rb') as f:
    dataset = pickle.load(f)

data = np.array(dataset['data'])
labels = np.array(dataset['labels'])
print(f"Loaded {len(data)} total landmark samples.")

#Filter classes
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

#Prepare data (Normalization was done in previous step)
X = data
y = labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
)

#Save encoder with matching name
with open('your_instance/mlp_label_encoder_84_img_aug.pickle', 'wb') as f:
    pickle.dump(label_encoder, f)
print("MLP LabelEncoder (84-feature, image augmented) saved.")

#Define the NEW 84-feature model
num_classes = y_train.shape[1]
model = Sequential([
    Dense(256, activation='relu', input_shape=(84,)), Dropout(0.5),
    Dense(128, activation='relu'), Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

#Train the model using model.fit
print("\nStarting model training...")
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
]
history = model.fit(X_train, y_train,
                    epochs=120, 
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=callbacks, 
                    verbose=1)

#Evaluate the final model (with potentially restored best weights)
print("\nEvaluating final model on the test set...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Accuracy: {accuracy * 100:.2f}%')
print(f'Test Loss: {loss:.4f}')


#Save model with matching name
model.save('your_instance/gesture_model_84_img_aug.h5')
print("\nNew 84-feature model (trained on image augmentations) successfully saved.")