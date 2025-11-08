import os
import requests
import streamlit as st
import tensorflow as tf

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model_url = "https://raw.githubusercontent.com/prasanthmpp2/animal_classifier/main/animal_classification_model.keras"
model_path = "animal_classification_model.keras"

if not os.path.exists(model_path):
    with open(model_path, "wb") as f:
        response = requests.get(model_url)
        f.write(response.content)

model = tf.keras.models.load_model(model_path)
