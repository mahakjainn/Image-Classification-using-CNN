from flask import Flask, request, render_template, redirect
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load your trained model
model = load_model('saved_model.keras')  # Ensure this path matches the saved model

# Define class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        image = Image.open(file).resize((32, 32))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        
        prediction = model.predict(image)
        class_index = np.argmax(prediction, axis=1)[0]
        predicted_class = class_names[class_index]
        
        return render_template('result.html', predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
