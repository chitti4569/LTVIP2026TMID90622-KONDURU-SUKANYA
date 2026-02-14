from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("healthy_vs_rotten.h5")

# Threshold: If confidence is below 80%, we say "Out of Dataset"
# Adjust this (0.0 to 1.0) based on your testing!
CONFIDENCE_THRESHOLD = 0.92  # Requires 92% certainty

classes = [
    'Apple__Healthy', 'Apple__Rotten', 'Banana__Healthy', 'Banana__Rotten',
    'Bellpepper__Healthy', 'Bellpepper__Rotten', 'Carrot__Healthy', 'Carrot__Rotten',
    'Cucumber__Healthy', 'Cucumber__Rotten', 'Grape__Healthy', 'Grape__Rotten',
    'Guava__Healthy', 'Guava__Rotten', 'Jujube__Healthy', 'Jujube__Rotten',
    'Mango__Healthy', 'Mango__Rotten', 'Orange__Healthy', 'Orange__Rotten',
    'Pomegranate__Healthy', 'Pomegranate__Rotten', 'Potato__Healthy', 'Potato__Rotten',
    'Strawberry__Healthy', 'Strawberry__Rotten', 'Tomato__Healthy', 'Tomato__Rotten'
]


@app.route("/")
def home():
    return render_template("index.html")


from tensorflow.keras.applications.vgg16 import preprocess_input # Add this import

@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    confidence = None
    image_path = None
    filename = None

    if request.method == "POST":
        file = request.files.get("pc_image")
        if file:
            filename = file.filename
            image_path = os.path.join("static/uploads", filename)
            os.makedirs("static/uploads", exist_ok=True)
            file.save(image_path)

            # ✅ 1. Standard VGG16 Preprocessing
            img = load_img(image_path, target_size=(224, 224))
            img_array = np.array(img)
            # If you used rescale=1/255 in training, keep the next line.
            # If not, comment it out and use preprocess_input(img_array)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # ✅ 2. Get All Probabilities
            probs = model.predict(img_array)[0]
            class_index = np.argmax(probs)
            max_prob = probs[class_index]

            # ✅ 3. Calculate "Confusion" (Gap between top 2 guesses)
            sorted_probs = np.sort(probs)
            top_choice = sorted_probs[-1]
            second_choice = sorted_probs[-2]
            gap = top_choice - second_choice

            # ✅ 4. STRICT FILTERING
            # A tiger shouldn't have a high gap, and its confidence shouldn't be massive.
            # We require at least 85% confidence AND a 40% gap over the next best guess.
            if top_choice < 0.85 or gap < 0.40:
                prediction = "Out of Dataset"
                confidence = top_choice * 100
            else:
                prediction = classes[class_index]
                confidence = top_choice * 100

    return render_template("portfolio-details.html", predict=prediction,
                           confidence=confidence, image_path=filename)


if __name__ == "__main__":
    app.run(debug=True, port=2222)