import SimpleITK as sitk
import numpy as np
import os

def registration(moving, fixed, scale=False, bspline=False):
	"""
	Registration: affine -> bspline(if turn on)
	Parameters:
	- moving: The image that need to be registered
	- fixed: The image that 'moving' registers to
	- scale: if True, scaled the registered image to range of 0 to 1
	- bspline: if True, use bspline after affine to better register
		- caution: bspline can greatly distort the image
	"""
	# initialize
	elastixImageFilter = sitk.ElastixImageFilter()
	elastixImageFilter.SetFixedImage(fixed)
	elastixImageFilter.SetMovingImage(moving)

	# set registration parameters
	parameterMap = sitk.VectorOfParameterMap()
	affine = sitk.GetDefaultParameterMap('affine')
	affine['AutomaticTransformInitialization'] = ['true']
	affine['MaximumNumberOfSamplingAttempts'] = ['10']
	affine['RequiredRatioOfValidSamples'] = ['0.05']

	# do nonlinear registration after affine if turn on
	parameterMap.append(affine)
	if bspline:
		parameterMap.append(sitk.GetDefaultParameterMap('bspline'))

	# start registration
	elastixImageFilter.SetParameterMap(parameterMap)
	elastixImageFilter.Execute()

	# scale the image if turn on
	reg = elastixImageFilter.GetResultImage()
	if scale:
		reg = reg / np.max(sitk.GetArrayFromImage(reg))
	return reg

def reg_tester():
	"""
	function tester
	"""
	moving = sitk.ReadImage('SUV_Rat5_2925_01mg_JB_32917.nii')
	fixed = sitk.ReadImage('PET_template.nii', sitk.sitkFloat32)
	reg = registration(moving, fixed, scale=False, bspline=False)
	sitk.WriteImage(reg, '../rats_reg/SUV_Rat5_2925_01mg_JB_32917.nii')
		
#reg_tester()
