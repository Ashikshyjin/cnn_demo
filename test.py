import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("image_classifier.h5")

class_names = ['bike', 'cars', 'cats', 'horses', 'flowers', 'dogs', 'human']

# Ask user for image path
image_path = input("Enter the image file path: ")

# Load image (unchanged image)
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

if image is None:
    print("Error: Could not load image.")
    exit()

# Convert RGBA -> RGB if image has 4 channels
if len(image.shape) == 3 and image.shape[2] == 4:
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

# Resize image
img = cv2.resize(image, (128, 128))

# Convert BGR -> RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Normalize
img = img / 255.0

# Add batch dimension
img = np.expand_dims(img, axis=0)

# Predict
predictions = model.predict(img)
class_index = np.argmax(predictions)
confidence = np.max(predictions)

label = f"{class_names[class_index]} ({confidence*100:.1f}%)"

# Display result
cv2.putText(image, label,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2)

cv2.imshow("Image Classification Result", image)

print("Prediction:", class_names[class_index])
print("Confidence:", confidence * 100)

cv2.waitKey(0)
cv2.destroyAllWindows()