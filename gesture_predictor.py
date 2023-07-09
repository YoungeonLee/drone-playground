import tensorflow as tf
import numpy as np
from utils import preprocess_data, IDX_TO_LABLES

class GesturePredictor:
    def __init__(self) -> None:
        self.model = tf.keras.models.load_model('gesture_recognizer.keras')
    
    def predict(self, data):
        if data.hand_landmarks:
            hand_landmarks = [[hand_landmark.x, hand_landmark.y, hand_landmark.z]
                            for hand_landmark in data.hand_landmarks[0]]
            processed_data = np.array(preprocess_data(hand_landmarks))

            predictions = self.model(np.array([processed_data])).numpy()[0] #type: ignore
            prediction = np.argmax(predictions)
            probability = predictions[prediction]

            return {'prediction': IDX_TO_LABLES[prediction], 'probability': probability}
        else:
            return None