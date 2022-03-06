from multiprocessing import context
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

import tensorflow_hub as hub
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import numpy as np
from tensorflow import Graph
import pickle

# Create your views here.

flower_list_path = "./models/102_flowers_list.pkl"
img_height, img_width = 299, 299

open_file = open(flower_list_path, "rb")
flower_species_list = pickle.load(open_file)
open_file.close()

model_graph = Graph()
with model_graph.as_default():
    gpuoptions = tf.compat.v1.GPUOptions(allow_growth=True)
    tf_session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpuoptions))
    with tf_session.as_default():
        cnn_model = load_model('./models/97.h5',custom_objects={'KerasLayer':hub.KerasLayer})



def index(request):
    context = {'a':1}
    return render(request, 'index.html', context)


def predictImage(request):
    imageFileObj = request.FILES['filePath']
    fs = FileSystemStorage()
    filePathName = fs.save(imageFileObj.name, imageFileObj)
    filePathName = fs.url(filePathName)

    testimage='.' + filePathName
    img = image.load_img(testimage, target_size=(img_height, img_width))
    x = image.img_to_array(img)
    x = x/255
    x = x.reshape(1,img_height, img_width, 3)

    with model_graph.as_default():
        with tf_session.as_default():
            prediction = cnn_model.predict(x)

    predictedLabel = flower_species_list[np.argmax(prediction)]

    context = {'filePathName': filePathName, 'predictedLabel': predictedLabel}
    return render(request, 'index.html', context)
