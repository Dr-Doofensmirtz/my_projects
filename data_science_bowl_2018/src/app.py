from flask import Flask, request, make_response, render_template
import os
import numpy as np

from PIL import Image
from skimage.io import imread
from skimage.transform import resize

from model import UNet

def segment(img):
    img = imread(image)[:,:,:3].astype('float32')
    img = resize(img, (128,128))
    img = np.expand_dims(img, axis=-1)

    out = model(img)
    out = out.


app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config['IMAGE_UPLOADs'] = os.path.join(APP_ROOT, 'static')

@app.route("/segmenter", methods = ['GET', 'POST'])
def predict():
    if request.method == "POST":
        image = request.files['input-files']
        filename = image.filename
        file_path = os.path.join(app.config['IMAGE_UPLOADs'], filename)
        image_pil = Image.open(image)
        image_pil.thumbnail((600,300), Image.ANTIALIAS)
        image_pil.save(file_path)

        

        out = segment(img)

