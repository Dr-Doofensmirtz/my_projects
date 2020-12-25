from flask import Flask, request, make_response, render_template
import os
import numpy as np

from PIL import Image
from skimage.io import imread, imsave
from skimage.transform import resize
import albumentations as A
from albumentations.pytorch import ToTensor
import torch

from model import UNet
import config

def predict(img_path, model):
    mean=(0.485, 0.456, 0.406) 
    std=(0.229, 0.224, 0.225)
    
    transform = A.Compose([
        A.Resize(128,128),
        A.Normalize(mean=mean, std=std),
        ToTensor()
    ])
    
    img = imread(img_path)[:,:,:3].astype('float32')
    augment = transform(image = img)
    img = augment['image']
    img = np.expand_dims(img, 0)

    out = model(torch.from_numpy(img)).cpu().detach().numpy()
    return np.squeeze(out, 0)[0]


app = Flask(__name__)
UPLOAD_FOLDER = "/home/atom/projects/data_science_bowl_2018/src/static"
PRED_PATH = "/home/atom/projects/data_science_bowl_2018/src/static/"
DEVICE = "cpu"


@app.route("/", methods = ['GET', 'POST'])
def upload_predict():
    if request.method == "POST":
        image_file = request.files['image']
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)
            pred = predict(image_location, MODEL)
            imsave(PRED_PATH + "pred.png", pred)
            return render_template("index.html", image_loc=image_file.filename, pred_loc = "pred.png")
    return render_template("index.html", prediction=0, image_loc=None)

        
if __name__ == "__main__":
    MODEL = UNet()
    MODEL.load_state_dict(torch.load(config.MODEL_LOAD_PATH))
    MODEL.to(DEVICE)
    MODEL.eval()
    app.run(port=12000, debug=True)