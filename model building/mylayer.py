"""
Self-defined layers and functions.
"""

import numpy as np
from math import sin, cos
from keras.engine.topology import Layer
from keras.callbacks import Callback
from keras.utils.np_utils import to_categorical
from scipy.ndimage.interpolation import affine_transform
from sklearn.preprocessing import scale
from sklearn.metrics import roc_auc_score
from keras import backend as K

def var(inputs):
	"""
	Variance layer: compute variance of two backends.
	"""
	from keras import backend as K
	x1, x2 = inputs
	return (K.square(x1) + K.square(x2))/2 - K.square((x1 + x2)/2)

def squeeze(input):
	"""
	Squeeze the empty dimension. (didn't use this function)
	"""
	from keras import backend as K
	return K.squeeze(input)

def zero_loss(y_true, y_pred):
	from keras import backend as K
	return K.zeros_like(y_pred)

class ConsistencyRegularization(Layer):
	"""
	Layer of consistency regularization
	"""
	def __init__(self, C=0.2, **kwargs):
		super(ConsistencyRegularization, self).__init__(**kwargs)
		self.C = C

	def call(self, x):
		from keras import backend as K
		in1, in2 = x
		self.res = self.C*K.sum((in1-in2)**2)
		self.add_loss(self.res, x)
		#you can output whatever you need, just update output_shape adequately
		#But this is probably useful
		return self.res

	def compute_output_shape(self, input_shape):
		return (input_shape[0][0], 1)


class roc(Callback):
	"""
	Compute ROC-AUC.
	"""
    def __init__(self,training_data, validation_data):
        self.x = training_data[0]
        self.y = to_categorical(training_data[1])
        self.x_val = validation_data[0]
        self.y_val = to_categorical(validation_data[1])

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))))
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

def roc_auc(y_ture, y_pred):
	return roc_auc_score(y_true, y_pred)

def augmentation(X, y, multiple=3, normalize=False, rotation=10, shift=0.1):
	"""
	Data augmentation for 3D images.
	Input:
	- multiple: int(>=1), number of new data multiple times of original
	- normalize: bool, feature-wise set mean to 0 and std to 1
	- rotation: int(0-180), range of random rotation(positive & negative)
	- shift: float(0-1), range of random shift(in all three direction)
	"""
	N, H, W, D = X.shape
	if N != y.size:
		print('Number of data and number of labels are not the same.')
		return 1
	X_aug = np.repeat(X, [multiple]*N, axis=0)
	y_aug = np.repeat(y, [multiple]*N, axis=0)

	for i in range(X_aug.shape[0]):
		# compute 3D transformation matrix
		T = np.array([H,W,D])*(np.random.rand(3)*shift*2 - shift)
		a, b, c = np.deg2rad(np.random.rand(3)*rotation*2 - rotation)
		Rz = np.array([[cos(a),-sin(a),0],[sin(a),cos(a),0],[0,0,1]])
		Ry = np.array([[cos(b),0,sin(b)],[0,1,0],[-sin(b),0,cos(b)]])
		Rx = np.array([[1,0,0],[0,cos(c),-sin(c)],[0,sin(c),cos(c)]])
		R = np.dot(np.dot(Rz, Ry), Rx)
		A = np.eye(4)
		A[:3,:3] = R
		A[:3,3] = T[:]
		offset = np.array([[1,0,0,H/2.+0.5],[0,1,0,W/2.+0.5],[0,0,1,D/2.+0.5],[0,0,0,1]])
		reset = np.array([[1,0,0,-(H/2.+0.5)],[0,1,0,-(W/2.+0.5)],[0,0,1,-(D/2.+0.5)],[0,0,0,1]])
		A = np.dot(np.dot(offset,A), reset)

		# apply transformation(translation and rotation)
		X_aug[i] = affine_transform(X_aug[i], A[:3,:3], A[:3,3], order=1)

	# normalize: center mean and unit variance: sample-wise -> feature-wise
	if normalize:
		X_aug = X_aug.reshape(X_aug.shape[0], -1)
		#X_aug = scale(X_aug, axis=1)
		X_aug = scale(X_aug, axis=0)
		X_aug = X_aug.reshape(X_aug.shape[0], H, W, D)

	return X_aug, y_aug

def tester():
	import SimpleITK as sitk
	img = sitk.ReadImage('data/test_rat.nii')
	X, y = augmentation(np.tile(sitk.GetArrayFromImage(img)[np.newaxis,...],(2,1,1,1)), np.array([0,1]), normalize=True)
	print(X[0])
	'''
	for i, x in enumerate(X):
		new = sitk.GetImageFromArray(x)
		new.CopyInformation(img)
		sitk.WriteImage(new, str(i)+'.nii')
	'''

#tester()
