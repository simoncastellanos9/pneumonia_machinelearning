import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		#print('upload_image filename: ' + filename)
		flash('Image successfully uploaded and displayed below')

		#-------------------------- PROCESSING UPLOADED X-RAY IMAGE
		# load the model
		model = load_model("input/chest_xray.h5")
		# model = load_model("input/chest_xray_30_Epochs.h5")

		# load and scale the image
		img=image.load_img(os.path.join(app.config['UPLOAD_FOLDER'], filename),target_size=(224,224))

		# convert image to array
		x=image.img_to_array(img)
		
		# reshape 
		x=np.expand_dims(x, axis=0)

		# preprocessing
		img_data=preprocess_input(x)

		# run model prediction
		classes=model.predict(img_data)

		# assign class based the prediction result
		result=int(classes[0][1])

		# if result==1:
		#     print("X-Ray results indicate Pneumonia")
		# else:
		#     print("X-Ray results are Normal")

		return render_template('upload.html', filename=filename, result=result)
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='img/' + filename), code=301)

if __name__ == "__main__":
    app.run()

# Code modified from https://roytuts.com/upload-and-display-image-using-python-flask/
