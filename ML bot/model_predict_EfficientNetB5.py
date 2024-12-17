import numpy as np
import tensorflow as tf
import cv2
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import io

model = tf.keras.models.load_model('my_model.h5')

def pred_disease(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = np.array(image)
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    img = cv2.resize(image, (180, 180))
    img = img / 255.0
    img = np.array([img])

    predict = model.predict(img)
    encod_pred = np.argmax(predict, axis=1)

    lb = LabelEncoder()
    lb.classes_ = np.array(['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella'])
    decoded_pred = lb.inverse_transform(encod_pred)

    return f"Predicted: {decoded_pred[0]}"


# if __name__ == "__main__":
#     with open("test.jpg", "rb") as f:  # Замените "test_image.jpg" на путь к вашему изображению
#         image_bytes = f.read()
#
#     result = pred_disease(image_bytes)
#     print(f"Predicted disease: {result}")