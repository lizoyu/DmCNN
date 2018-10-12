"""
This script is for building the MRI-only model based on DmCNN.
Model pipeline:
MRI - MRI branch - GAP - softmax
"""
from keras.layers import Dense
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model, load_model
from keras import backend as K
from mylayer import var

# load DmCNN model
dmcnn = load_model('models/dmcnn_pretrained.h5')

# build the MRI-only model and save
x = GlobalAveragePooling2D()(dmcnn.layers[208-1].output)
prediction = Dense(2, activation='softmax', use_bias=False)(x)
mri_only = Model(inputs=dmcnn.input[1], outputs=prediction)

mri_only.save('models/mrionly_pretrained.h5')
