from flask import Flask, render_template, url_for, request,redirect
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import joblib 
from tensorflow.keras.models import load_model
model = load_model('skin_cancer_model.h5')
std = joblib.load('StandardScaler.lb')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict",methods = ('GET','POST'))
def predict():
        if 'image' not in request.files:
            return 'No file part'
        Gender = request.form['gender']
        dx_type = request.form['dx_type']
        localization = request.form['localization']
        age = request.form['age']
        feature = [age,Gender,localization,dx_type]
        features = [feature]
        # Get upload image 
        file = request.files['image']
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img_path = os.path.join(app.config['UPLOAD_FOLDER'],filename)
        img = image.load_img(img_path, target_size=(224, 224))  # Adjust target_size as needed
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)  # Convert to a batch of size 1
          
        # Create an instance of ImageDataGenerator with preprocessing
        test_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rescale=1./255  # Optionally rescale pixel values
        )

        # Flow the single image through the datagen to apply preprocessing
        img_gen = test_datagen.flow(x, batch_size=1)
        preprocessed_img = next(img_gen)
        std_feature = std.transform(features)
        prediction = model.predict([preprocessed_img,std_feature])
        predicted_classes = np.argmax(prediction, axis=1)
        class_name = ['mel', 'nv',  'bkl',"df, vasc, akiec, bcc"]
        predict_class = class_name[predicted_classes[0]]
        return render_template("result.html", prediction = predict_class)
    

if __name__=="__main__":
    app.run(debug=True)