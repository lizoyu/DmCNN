"""
This script is for building the PET-only model based on DmCNN.
Model pipeline:
PET - PET branch - GAP - softmax
"""
from keras.layers import Dense
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model, load_model
from keras import backend as K
from mylayer import var

# load DmCNN model
dmcnn = load_model('models/dmcnn_pretrained.h5')

# build the PET-only model and save
x = GlobalAveragePooling2D()(dmcnn.layers[207-1].output)
prediction = Dense(2, activation='softmax', use_bias=False)(x)
pet_only = Model(inputs=dmcnn.input[0], outputs=prediction)

pet_only.save('models/petonly_pretrained.h5')
