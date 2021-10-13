import warnings
warnings.filterwarnings('ignore')
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.models import load_model
import warnings

model = load_model('../input/chest_xray.h5')

path = '../input/labeled-chest-xray-images/chest_xray/test/PNEUMONIA/'

file = 'person21_virus_53.jpeg'

img = image.load_img(path+file, target_size=(224, 224))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

img_data = preprocess_input(x)
classes = model.predict(img_data)
result = int(classes[0][0])

if result == 0:
  print("X-Ray results indicate Pneumonia")
else:
  print("X-Ray results are Normal")
