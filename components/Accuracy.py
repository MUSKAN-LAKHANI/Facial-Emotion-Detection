import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data from CSV file
data = pd.read_csv('fer2013.csv')

# Split features (pixels) and labels (emotions)
pixels = data['pixels'].values
labels = data['emotion'].values

# Convert pixels to numpy array
pixels = np.array([np.fromstring(pixel, dtype=int, sep=' ') for pixel in pixels])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(pixels, labels, test_size=0.2, random_state=42)

# Assuming you have already trained your model and made predictions on the test set
# Replace predictions with your actual predictions
# For demonstration purposes, let's assume predictions are random
predictions = np.random.randint(0, 7, size=len(y_test))  # Random predictions

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

# Plot accuracy graph
print(f"Accuracy: {accuracy}")
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, 'b', label='Accuracy')
plt.title('Accuracy Over Time')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()