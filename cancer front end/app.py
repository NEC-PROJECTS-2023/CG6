# import os
# from flask import Flask
# from flask import request
# from flask import render_template
# import numpy as np
# app = Flask(__name__) 
# UPLOAD_FLODER = "Downloads\cancer"
# def ValuePredictor(to_predict_list):
# 	to_predict = np.array(to_predict_list).reshape(1, 12)
# 	loaded_model = (open("weights.best.hdf5", "rb"))
# 	result = loaded_model.predict(to_predict)
# 	return result[0]

# @app.route("/", methods=["GET" , "POST"])
# def upload_predict():
#     if request.method == "POST":
#          image_file = request.files["image"]
#          if image_file:
#              image_location = os.path.join(
#                  UPLOAD_FLODER,
#                  image_file.filename
        
#              )
#              image_file.save(image_location)
#              return render_template("index.html", prediction=1)
#     return render_template("index.html", prediction=0)

# if __name__ == "__main__":
#     app.run(port=12000, debug=True)

from __future__ import division, print_function
import os
import numpy as np
# Keras
import tensorflow as tf
from PIL import Image
import cv2
# Flask utils
from flask import Flask,  url_for, request, render_template,send_from_directory
from werkzeug.utils import secure_filename
from heatmap import save_and_display_gradcam,make_gradcam_heatmap



os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# Define a flask app
app = Flask(__name__, static_url_path='')


app.config['HEATMAP_FOLDER'] = 'heatmap'
app.config['UPLOAD_FOLDER'] = 'uploads'
# Model saved with Keras model.save()
# MODEL_PATH = 'E:\cancer\models\weights.best (1).hdf5'


# #Load your trained model
# model = load_model(MODEL_PATH)
#         # Necessary to make everything ready to run on the GPU ahead of time
# print('Model loaded. Start serving...')


class_dict = {0:"Benign (Cancer)",
             1:"Melanoma (Cancer)"}

@app.route('/uploads/<filename>')
def upload_img(filename):
    
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
        
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')



def model_predict(img_path, model):
    
    img = Image.open(img_path).resize((224,224)) #target_size must agree with what the trained model expects!!

    # Preprocessing the image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32')/255
   
    preds = model.predict(img)[0]
    prediction = sorted(
      [(class_dict[i], round(j*100, 2)) for i, j in enumerate(preds)],
      reverse=True,
      key=lambda x: x[1]
  )
    return prediction,img


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        print(file_path)
        f.save(file_path)
        file_name=os.path.basename(file_path)
        # Make prediction
        model_json = open('./models/model.json', 'r')
        loaded_model_json = model_json.read()
        model = tf.keras.models.model_from_json(loaded_model_json)
        model.load_weights("./models/model.h5")
        def image_loader(PATH, RESIZE, sigmaX=10):
            read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
        
            
            
            img = read(PATH)
                
            img = cv2.resize(img, (RESIZE,RESIZE))
                

            return img
        a = image_loader(file_path,224)
        y = model.predict(np.reshape(a,(-1,224,224,3)))
        if y[0,0]<y[0,1]:
             pred = "Malignant"
        else:
             pred= "Benign"
        # print(y.shape)
        # print(type(y))
        
    return render_template('classify.html',result=pred)




    #this section is used by gunicorn to serve the app on Azure and localhost
if __name__ == '__main__':
        app.run(debug=True, host="localhost", port=8080)