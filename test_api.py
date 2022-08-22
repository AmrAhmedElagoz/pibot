# import requests

# url = 'http://127.0.0.1:5000/chat'
# r = requests.post(url,json={"_title":"this is test title"})

# print(r.json())
import warnings
from tag_classification import send
warnings.filterwarnings('ignore')

import pickle
import tensorflow as tf
from keras.models import load_model
# my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.set_visible_devices([], 'GPU')

model = load_model('chatbot_model_v2.h5')
print(model)

send()