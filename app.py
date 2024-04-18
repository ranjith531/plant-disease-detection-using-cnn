
from __future__ import division, print_function
from werkzeug.utils import secure_filename
from flask import Flask, redirect, url_for, request, render_template
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
# coding=utf-8
import sys
import os
import glob
import re
import numpy as nps
# import tensorflow as tf
import tensorflow as tf


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras

# Flask utils
# from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__, template_folder='templates')

# Model saved with Keras model.save()
MODEL_PATH = 'plant.h5'

# Load your trained model
model = load_model(MODEL_PATH)


def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(256, 256))

    # Preprocessing the image
    x = image.img_to_array(img)  # [25*256]
    # x = np.true_divide(x, 255)
    # Scaling
    x = x/255
    x = nps.expand_dims(x, axis=0)  # np

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # x = preprocess_input(x)

    preds = model.predict(x)
    preds = nps.argmax(preds, axis=1)  # np
    if preds == 0:
        preds1 = "Plant Name : Pepper bell "
        preds2 = "Disease : Bacterial spot "
        preds3 = "Treatment : Remove infected parts and Use a copper based fungicide as a foliar spray in the early morning or late evening to help reduce the spread."
        preds4 = "Fertilizers: Copper Hydroxide"
    elif preds == 1:
        preds1 = "Plant Name : Pepper bell"
        preds2 = "No Disease : healthy "
        preds3 = "Treatment: No need to use any fertilizer as the plant is healthy"
        preds4 = "Fertilizers : No need"

    elif preds == 2:
        preds1 = "Plant Name : Potato "
        preds2 = "Disease : Early blight "
        preds3 = "Treatment: Treat early blight by removing lower leaves, including up to one-third of infected foliage. Apply a tomato fungicide at the first sign of infection or when weather conditions are favorable for disease to develop. Prevent early blight by watering at soil level and mulching"
        preds4 = "Fertilizers: Azoxystrobin ,Boscalid, Chlorothalonil ,Famoxadone ,Cymoxanil"

    elif preds == 3:
        preds1 = "Plant Name : Potato "
        preds2 = "Disease : Late blight "
        preds3 = "Treatment:Only disease-free potatoes should be used for seeds. Potato dumps or cull piles should be burned before planting time or sprayed with strong herbicides to kill all sprouts or green growth."
        preds4 = "Fertilizers: Azoxystrobin ,Cymoxanil,Chlorothalonil "
    elif preds == 4:
        preds = "Plant Name : Potato "
        preds2 = "No Disease : healthy |"
        preds3 = "Treatment : No treatment required"
        preds4 = "Fertilizers: No need to use any fertilizer "
    elif preds == 5:
        preds1 = "Plant Name : Tomato "
        preds2 = "Disease : Bacterial spot "
        preds3 = "Treatment:Use copper hydroxide to prevent Bacterial spot."
        preds4 = "Fertilizers: Copper Hydroxide , Copper plus Mancozeb"

    elif preds == 6:
        preds1 = "Plant Name : Tomato "
        preds2 = "Disease : Early blight "
        preds3 = "Treatment:Use copper Fungicides to prevent early blight disease"
        preds4 = "Fertilizers: Azoxystrobin, Pyraclostrobin, Difenoconazole, Boscalid, Chlorothalonil, Fenamidone, Maneb, Mancozeb, Trifloxystrobin, and Ziram."
    elif preds == 7:
        preds1 = "Plant Name : Tomato "
        preds2 = "Disease : Late blight "
        preds3 = "Treatment: Avoid sprinkler system if possible because it favours the development of late blight"
        preds4 = "Fertilizers: Fdimethomorph, Azoxystrobin, Azoxystrobin +Difenoconazole, Mancozeb, Amoxadone + Cymoxanil,"
    elif preds == 8:
        preds1 = "Plant Name : Tomato "
        preds2 = "Disease : Leaf Mold "
        preds3 = "Treatment :Use fungicidal sprays like Calcium chloride-based sprays are recommended for treating leaf mold issues. Organic fungicide options are also available."
        preds4 = "Fertilizers: Chlorothalonil, Maneb, Mancozeb and Copper"
    elif preds == 9:
        preds1 = "Plant Name : Tomato "
        preds2 = "Disease : Septoria leaf spot "
        preds3 = "Treatment: Cage tomatoes to prevent leaves from touching the ground or other plants. Water aids the spread of Septoria leaf spot. Keep it off the leaves as much as possible by watering at the base of the plant only. Of course, it's impossible to keep the rain off your plants, but every little bit helps."
        preds4 = "Fertilizers: Azoxystrobin, Penthiopyrad, Potassium Bicarbonate"
    elif preds == 10:
        preds1 = "Plant Name : Tomato "
        preds2 = "Disease : Spider mites Two spotted spider mite "
        preds3 = "Treatment: Spider mites can be treated with neem oil, insecticide sprays, or diatomaceous earth. But if they get too out of control they’ll likely kill your plants. Your best bet is to try to prevent them by spraying seaweed extract weekly."
        preds4 = "Fertilizers: BotaniGard ES,Nuke Em, Insecticidal soaps or Botanical Insecticides"
    elif preds == 12:
        preds1 = "Plant Name : Tomato "
        preds2 = "Disease : Target Spot "
        preds3 = "Treatment: Warm wet conditions favour the disease such that fungicides are needed to give adequate control. The products to use are chlorothalonil, copper oxychloride or mancozeb. Treatment should start when the first spots are seen and continue at 10-14-day intervals until 3-4 weeks before last harvest."
        preds4 = "Fertilizers: Chlorothalonil, Mancozeb, and Copper Oxychloride"
    elif preds == 13:
        preds1 = "Plant Name : Tomato "
        preds2 = "Disease : Yellow Leaf Curl Virus "
        preds3 = "Treatment: Plant immediately after any tomato-free period or true winter season. Avoid planting new fields near older fields (especially those with TYLCV-infected plants) Manage WHITEFLIES. "
        preds4 = "Fertilizers: Spray with Azadirachtin (Neem), Pyrethrin or Insecticidal soap "
    elif preds == 14:
        preds1 = "Plant Name : Tomato "
        preds2 = "Disease : Mosaic Virus "
        preds3 = "Treatment: mosaic virus is difficult and there are no chemical controls like there are for fungal diseases, although some varieties of tomato are resistant to the disease, and seeds can be bought that are certified disease free."
        preds4 = "Fertilizers: The use of silver reflective mulches may delay the infection so use silver compounds"
    else:
        preds1 = "Plant Name : Tomato "
        preds2 = "No Disease : healthy "
        preds3 = "Treatment: plant is healthy so treatment is not required"
        preds4 = "Fertilizers: No Need"
    output1 = preds1
    output2 = preds2
    output3 = preds3
    output4 = preds4
    return render_template('result.html', output1=output1, output2=output2, output3=output3, output4=output4)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template("index.html")


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']  # xyz

        # Save the file to ./uploads
        basepath = os.path.dirname("uploads")
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))  # plant/uploads/xyz.jpg
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result = preds
        return result
    return None


if __name__ == '__main__':
    app.run(port=5001, debug=True)
