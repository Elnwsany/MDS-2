from flask import Flask, render_template, request
from keras.models import load_model
import cv2
import pandas as pd

rec_app = Flask(__name__)


@rec_app.route("/")
def Home():

    return render_template("index.html")


@rec_app.route("/index.html")
def returnHome():

    return render_template("index.html")


@rec_app.route("/About.html")
def About():

    return render_template("About.html")


@rec_app.route("/Contact.html")
def Contact():

    return render_template("Contact.html")


@rec_app.route("/Chest.html", methods=['GET', 'POST'])
def Chest():
    image_path = 0
    x = 0
    if request.method == 'POST':
        imagefile = request.files['imagefile']
        image_path = "./images/" + imagefile.filename
        imagefile.save(image_path)

    return render_template("Chest.html", path=image_path)


@rec_app.route("/covid.html", methods=['GET', 'POST'])
def Covid():
    image_path = 0
    x = 0
    if request.method == 'POST':
        imagefile = request.files['imagefile']
        image_path = "./images/" + imagefile.filename
        imagefile.save(image_path)

        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        # print(img.shape)
        img = img/255
        model = load_model('covid_best_model.h5')
        pred = model.predict(img.reshape(1, 224, 224, 3), batch_size=1)
        # print(pred)
        pred = pred.argmax()
        # print(pred)
        if pred == 0:
            x = 'COVID-19'
        elif pred == 1:
            x = 'Normal'
        else:
            x = 'Pneumonia'

    return render_template("covid.html", path=x)


@rec_app.route("/Heartbeat.html", methods=['GET', 'POST'])
def Heartbeat():
    image_path = 0
    x = 0
    if request.method == 'POST':
        imagefile = request.files['imagefile']
        image_path = "./images/" + imagefile.filename
        imagefile.save(image_path)

        ecg_sig = pd.read_csv(image_path)
        ecg_sig = ecg_sig.values
        ecg_sig = ecg_sig.reshape(1, 187, 1)
        model = load_model('ecg_data_best_model.h5')
        pred = model.predict(ecg_sig, batch_size=1)
        pred = pred.argmax()

        if pred == 0:
            x = 'Normal Beat'
        elif pred == 1:
            x = 'Supraventricular premature beat '
        elif pred == 2:
            x = 'Premature ventricular contraction'
        elif pred == 3:
            x = 'Fusion of ventricular'
        else:
            x = 'Unknown beat'

    return render_template("Heartbeat.html", path=x)


@rec_app.route("/Brain.html", methods=['GET', 'POST'])
def braintumor():
    image_path = 0
    x = 0
    if request.method == 'POST':
        imagefile = request.files['imagefile']
        image_path = "./images/" + imagefile.filename
        imagefile.save(image_path)

        print(image_path)
        img = cv2.imread(image_path)
        model = load_model('brain_best_model.h5')
        img = cv2.resize(img, (200, 200))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img/255
        pred = model.predict(img.reshape(1, 200, 200, 1), batch_size=1)
        print(pred)
        pred = pred.argmax()
        print(pred)
        if pred == 0:
            x = 'glioma'
        elif pred == 1:
            x = 'meningioma'
        elif pred == 2:
            x = 'no tumor'
        else:
            x = 'pituitary'

    return render_template("Brain.html", path=x)


if __name__ == "__main__":
    rec_app.run(debug=True, port=9000)
