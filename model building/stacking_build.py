"""
This script is for building the model that stacks MRI and PET as input.
Model pipeline:
MRI \
      - concat - MRI branch - GAP - softmax
PET /
"""
from keras.layers import Conv2D, MaxPooling2D, Input, Activation, Add, Average, ZeroPadding2D, Lambda
from keras.layers import UpSampling2D, AveragePooling2D, Concatenate, LeakyReLU, SpatialDropout2D
from keras.layers import BatchNormalization, Flatten, Dense
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model, load_model
from keras import backend as K
from metrics import weighted_dice_coefficient_loss
from mylayer import var
import numpy as np

# load MRI-only model
mri_only = load_model('models/mrionly_pretrained.h5')

# build a new model
stack_inputs = Input(shape=(120,144,240), name='stack_input_1')
x = Conv2D(64, (7,7), padding='same', use_bias=False)(stack_inputs)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = ZeroPadding2D()(x)
connect_point = MaxPooling2D((3,3), strides=(2,2))(x)
##### Dense Block 1
for _ in range(6):
    x = BatchNormalization()(connect_point)
    x = LeakyReLU()(x)
    x = Conv2D(128, (1,1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(32, (3,3), padding='same', use_bias=False)(x)
    connect_point = Concatenate()([connect_point,x])
##### Transition Block 1
x = BatchNormalization()(connect_point)
x = LeakyReLU()(x)
x = Conv2D(128, (1,1), padding='same', use_bias=False)(x)
connect_point = AveragePooling2D()(x)
##### Dense Block 2
for _ in range(12):
    x = BatchNormalization()(connect_point)
    x = LeakyReLU()(x)
    x = Conv2D(128, (1,1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(32, (3,3), padding='same', use_bias=False)(x)
    connect_point = Concatenate()([connect_point,x])
##### Transition Block 2
x = BatchNormalization()(connect_point)
x = LeakyReLU()(x)
x = Conv2D(256, (1,1), padding='same', use_bias=False)(x)
x = GlobalAveragePooling2D()(x)
prediction = Dense(2, activation='softmax', use_bias=False)(x)

stacking = Model(inputs=stack_inputs, outputs=prediction)

# fill in the weights
stacking.layers[1].set_weights([np.tile(mri_only.layers[1].get_weights()[0], (1,1,2,1))])
print(stacking.layers[1].get_config()['name'], mri_only.layers[1].get_config()['name'])

for i, j in zip(range(2, len(stacking.layers)), range(2, len(mri_only.layers))):
    if stacking.layers[i].get_weights():
        stacking.layers[i].set_weights(mri_only.layers[j].get_weights())
        print(stacking.layers[i].get_config()['name'], mri_only.layers[j].get_config()['name'])

# save the model
stacking.save('models/stacking_pretrained.h5')