import tensorflow as tf
import numpy as np
from PIL import Image
import io

model = tf.keras.models.load_model(
    'efficientnetb3.h5',
    custom_objects={'TFOpLambda': tf.identity}
)

class_labels = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']

def predict_from_bytes(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions[0])

    return class_labels[predicted_class]

if __name__ == "__main__":
    with open("test.jpg", "rb") as f:
        image_bytes = f.read()

    result = predict_from_bytes(image_bytes)
    print(f"Predicted disease: {result}")
