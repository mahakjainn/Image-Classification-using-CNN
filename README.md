# Image-Classification-using-CNN
Created a Machine Learning model for Image Classification using CNN and deployed it with a user friendly web interface using Flask .








Objectives:
Develop a CNN-based image classification model.
Deploy the model with a web interface for user interaction.

Dataset:
Source: CIFAR-10 dataset (60,000 32x32 color images in 10 classes)

Project Structure:
imageClassificationProject.ipynb: Jupyter Notebook with code for data preprocessing, model training, and evaluation.
app.py: Flask application script for deployment.
static/ and templates/: Directories for web app static files and HTML templates.
saved_model.keras: Trained model file.

Setup Instructions:
Prerequisites
Python 3.x
Required packages (install via requirements.txt)


Installation:

Clone the Repository:
git clone https://github.com/yourusername/image-classification-cnn.git
cd image-classification-cnn

Create Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install Dependencies:
pip install -r requirements.txt
Running the Jupyter Notebook

Start Jupyter Notebook:
jupyter notebook
Open and Run imageClassificationProject.ipynb

Running the Flask App:
Ensure saved_model.keras is present.

Start the Flask Server:
python app.py

Access the Web Interface:
Go to http://127.0.0.1:5000/ in your browser.
