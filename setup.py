import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle

# Load the dataset
file_path = "Career Dataset.xlsx"  # Update the path if needed
df = pd.read_excel(file_path)

# One-Hot Encode Skills
df_skills = df["Skill"].str.get_dummies(sep=", ")

# Combine with Career column
df_processed = pd.concat([df["Career"], df_skills], axis=1)

# Save processed data
df_processed.to_csv("processed_career_dataset.csv", index=False)

# Display the first few rows
df_processed.head()

# Load the processed dataset
df = pd.read_csv("processed_career_dataset.csv")

# Separate features (X) and target (y)
X = df.drop(columns=["Career"])  # Skills as features
y = df["Career"]  # Career as target

# Encode the target labels (convert career names to numbers)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define the Neural Network
model = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dense(len(label_encoder.classes_), activation="softmax")  # Output layer for classification
])

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))


# Save the trained model
model.save("career_nn_model.h5")

# Test loading the file
try:
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    print("Label encoder loaded successfully!")
except Exception as e:
    print(f"Error loading label_encoder.pkl: {e}")

# Load model and label encoder
model = keras.models.load_model("career_nn_model.h5")

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Example: Predict Career from New Skills
import numpy as np
new_skills = np.array([X.iloc[0]])  # Replace with actual skill inputs
prediction = model.predict(new_skills)
predicted_career = label_encoder.inverse_transform([np.argmax(prediction)])

print("Predicted Career:", predicted_career[0])



import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, classification_report

# Load the trained model
model = keras.models.load_model("career_nn_model.h5")

# Make predictions on the test set
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

# Print results
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_rep)

