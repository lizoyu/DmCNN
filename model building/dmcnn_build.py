"""
This script is for building the DmCNN and fill in the pre-trained weights.
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

K.clear_session()

"""
Modify the pretrained model path here.
"""
denset_path = 'models/densenet.h5'
fcn_path = 'models/isensee_2017_model.h5'

"""
Build the DmCNN model from scratch.
Model pipeline:
    PET branch \
                - Merge module -> backend -> prediction
    MRI branch /
"""
# Frontend: PET branch
pet_inputs = Input(shape=(120,144,120), name='pet_input_1')
# level 1: down
x = Conv2D(16, (3,3), padding='same', name='lv1_down_conv_1')(pet_inputs) # 120,144,120
x = BatchNormalization(name='lv1_down_bn_1')(x)
res = LeakyReLU(name='lv1_down_lrelu_1')(x)
##### context module 1
x = Conv2D(16, (3,3), padding='same', name='lv1_down_conv_2')(res)
x = BatchNormalization(name='lv1_down_bn_2')(x)
x = LeakyReLU(name='lv1_down_lrelu_2')(x)
x = SpatialDropout2D(0.3, name='lv1_down_dropout_1')(x)
x = Conv2D(16, (3,3), padding='same', name='lv1_down_conv_3')(x)
x = BatchNormalization(name='lv1_down_bn_3')(x)
x = LeakyReLU(name='lv1_down_lrelu_3')(x)
down1 = Add(name='lv1_down_add_1')([res,x]) # 120,144,120
# level 2: down
x = ZeroPadding2D(name='lv2_down_zeropad_1')(down1)
x = Conv2D(32, (3,3), strides=(2,2), padding='valid', name='lv2_down_conv_1')(x) # 60,72,60
x = BatchNormalization(name='lv2_down_bn_1')(x)
res = LeakyReLU(name='lv2_down_lrelu_1')(x)
##### context module 2
x = Conv2D(32, (3,3), padding='same', name='lv2_down_conv_2')(res)
x = BatchNormalization(name='lv2_down_bn_2')(x)
x = LeakyReLU(name='lv2_down_lrelu_2')(x)
x = SpatialDropout2D(0.3, name='lv2_down_dropout_1')(x)
x = Conv2D(32, (3,3), padding='same', name='lv2_down_conv_3')(x)
x = BatchNormalization(name='lv2_down_bn_3')(x)
x = LeakyReLU(name='lv2_down_lrelu_3')(x)
down2 = Add(name='lv2_down_add_1')([res,x]) # 60,72,60
#level 3: down
x = ZeroPadding2D()(down2)
x = Conv2D(64, (3,3), strides=(2,2), padding='valid')(x) # 30,36,30
x = BatchNormalization()(x)
res = LeakyReLU()(x)
##### context module 3
x = Conv2D(64, (3,3), padding='same')(res)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = SpatialDropout2D(0.3)(x)
x = Conv2D(64, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Add()([res,x])
##### up
x = UpSampling2D()(x) # 60,72,60
x = Conv2D(32, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
# level 2: up
x = Concatenate()([down2,x]) # 60,72,60
x = Conv2D(32, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Conv2D(32, (1,1), padding='same')(x)
x = BatchNormalization()(x)
up2 = LeakyReLU()(x)
x = UpSampling2D()(up2) # 120,144,120
x = Conv2D(16, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
# level 1: up
x = Concatenate()([down1,x])
x = Conv2D(16, (3,3), padding='same')(x)
x = BatchNormalization()(x)
up1 = LeakyReLU()(x)
##### deep supervision path
x = Conv2D(16, (1,1), padding='same')(up2)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = UpSampling2D()(x) # 120,144,120
x = Add()([up1,x])
# Transition Block
x = Conv2D(1, (1,1), padding='valid')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Conv2D(128, (1,1), padding='same', use_bias=False)(x)
x = AveragePooling2D()(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Conv2D(256, (1,1), padding='same', use_bias=False)(x)
pet_end = AveragePooling2D()(x)

# Frontend: MRI branch
mri_inputs = Input(shape=(120,144,120))
x = Conv2D(64, (7,7), padding='same', use_bias=False)(mri_inputs)
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
mri_end = Conv2D(256, (1,1), padding='same', use_bias=False)(x)

# Merge module
mean = Average()([pet_end,mri_end])
var = Lambda(var)([pet_end,mri_end]) # customized
x = Concatenate()([mean,var])

# Backend: DenseNet
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Conv2D(64, (1,1), padding='same', use_bias=False)(x)
connect_point = AveragePooling2D()(x)
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
# Final Prediction
x = BatchNormalization()(connect_point)
x = LeakyReLU()(x)
x = Conv2D(2, (1,1), padding='valid')(x)
x = GlobalAveragePooling2D()(x)
prediction = Activation('softmax')(x)

# form the model
dmcnn = Model(inputs=[pet_inputs,mri_inputs], outputs=prediction)

"""
Fill in the pre-trained weights.
"""
# load the pretrained model
densenet = load_model('models/densenet.h5')
print('DenseNet loaded.')
fcn = load_model('models/isensee_2017_model.h5', 
        custom_objects={'weighted_dice_coefficient_loss': weighted_dice_coefficient_loss})
print('Residual FCN loaded.')

# MRI branch
dmcnn.layers[1].set_weights([np.tile(densenet.layers[2].get_weights()[0], (1,1,40,1))])
print(dmcnn.layers[1].get_config()['name'], densenet.layers[2].get_config()['name'])

layer_idx = [(8,9), (11,12), (15,16), (18,19), (22,23), (25,26), (29,30), (32,33), (36,37), (39,40), (43,44)]
layer_idx += [(46,47), (50,51), (54,55), (57,58), (61,62), (64,65), (68,69), (71,72), (78-1,76), (84-1,79), (92-1,83)]
layer_idx += [(98-1,86), (106-1,90), (112-1,93), (120-1,97), (126-1,100), (134-1,104), (140-1,107), (148-1,111), (154-1,114)]
layer_idx += [(162-1,118), (168-1,121), (177-2,125), (186-2,128), (194-1,132), (200-1,135), (208-1,139)]
for i, j in layer_idx:
    dmcnn.layers[i].set_weights(densenet.layers[j].get_weights())
    print(dmcnn.layers[i].get_config()['name'], densenet.layers[j].get_config()['name'])

# PET branch
dmcnn.layers[75].set_weights([fcn.layers[1].get_weights()[0][:,:,:,:1,:], fcn.layers[1].get_weights()[1]])
print(dmcnn.layers[75].get_config()['name'], fcn.layers[1].get_config()['name'])

layer_idx = [(81,4), (89,8), (99,12), (105,15), (113,19), (123,23), (129,26), (137,30), (147,79), (155,83), (161,86)]
layer_idx += [(169,90), (178,94)]

for i, j in layer_idx:
    dmcnn.layers[i].set_weights(fcn.layers[j].get_weights())
    print(i, dmcnn.layers[i].get_config()['name'], j, fcn.layers[j].get_config()['name'])

#### transition block using DenseNet
dmcnn.layers[197].set_weights([densenet.layers[51].get_weights()[0][:,:,:120,:]])
print(dmcnn.layers[197].get_config()['name'], densenet.layers[51].get_config()['name'])
dmcnn.layers[205].set_weights([densenet.layers[139].get_weights()[0][:,:,:128,:]])
print(dmcnn.layers[205].get_config()['name'], densenet.layers[139].get_config()['name'])

# frontend
dmcnn.layers[214-1].set_weights([densenet.layers[139].get_weights()[0][...,:64]])
print(dmcnn.layers[214].get_config()['name'], densenet.layers[139].get_config()['name'])

i = 218-1; j = 9; isFour = False
while i <= 256:
    dmcnn.layers[i].set_weights(densenet.layers[j].get_weights())
    print(i, dmcnn.layers[i].get_config()['name'], j, densenet.layers[j].get_config()['name'])
    if isFour:
        inc = 4
        isFour = False
    else:
        inc = 3
        isFour = True
    i += inc
    j += inc
    
dmcnn.layers[260-1].set_weights(densenet.layers[51].get_weights())
print(dmcnn.layers[260-1].get_config()['name'], densenet.layers[51].get_config()['name'])

i = 264-1; j = 55; isFour = False
while i <= 344:
    dmcnn.layers[i].set_weights(densenet.layers[j].get_weights())
    print(dmcnn.layers[i].get_config()['name'], densenet.layers[j].get_config()['name'])
    if isFour:
        inc = 4
        isFour = False
    else:
        inc = 3
        isFour = True
    i += inc
    j += inc

# save the final DmCNN model
dmcnn.save('models/dmcnn_pretrained.h5')