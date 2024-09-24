from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model('cyberguard_model.h5')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    prediction = model.predict(np.array(data).reshape(1, -1))
    return jsonify({'prediction': int(prediction[0] > 0.5)})

if __name__ == '__main__':
    app.run(debug=True)
