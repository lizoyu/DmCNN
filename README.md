# DmCNN: Dual-modal Convolutional Networks For Prediction of Progression of Mild Cognitive Impairment into Alzheimer's Disease

This is a self-motivated research project aimed at the prediction of the progression of Mild Cognitive Impairment (MCI) into Alzheimer's Disease (AD) using deep learning techniques.

## Background

Mild Cognitive Impairment (MCI) is defined as an impairment of cognitive function severer than the cognitive decline due to aging, while not severe enough to be classified as Alzheimer's Disease (AD) or dementia. Subjects with MCI are more likely to progress into AD, as progressive-MCI (p-MCI), but also possible to remain stable, as stable-MCI (s-MCI), or even revert to normal state.

A common concern for patients with MCI and their relatives is the likelihood and possible time of progression to AD or eventually dementia, which generally is 5% to 10% per year but also depends on various factors, such as social environment, diet, education.

Structural Magnetic Resonance Imaging (MRI) is an imaging modality that produces detailed and relatively high-resolution images of internal body structures, and is extensively used in this task. Another imaging technique, fluorodeoxyglucose(FDG)-positron emission tomography (PET), which observes metabolic processes in the body with the aid of FDG, a radiopharmaceutical, is also used for prediction of MCI progression to AD.

While PET images are less used for sMCI/pMCI classication alone, probably due to less available data and lower resolution, several studies suggests that FDG-PET performs better for patients with early-onset MCI, while MRI outperforms in late-onset MCI subjects, which conforms with the characteristic of the two modalities. MRI generates internal brain structure, thus is able to indicate structural deformation, while FDG-PET shows brain metabolism, indicating possible metabolic abnormalities. When brain structure deforms or degenerates, it usually infers the late period of a disease, while abnormality in brain metabolism usually occur earlier than structure changes. Therefore, it is a good idea to take advantage of both modalities to predict the risk of AD progression of MCI patients.

Convolutional Neural Networks(CNN), the neural network model based on 2D or 3D convolutions, are
predominant in a variety of image-related areas and gaining popularity in medical image computing field. Several studies had applied CNN for AD progression of MCI patients.

## Goal

Classification of sMCI and pMCI using MRI and PET images

## Data

Data used in the preparation of this article were obtained from the Alzheimers Disease Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). The ADNI was launched in 2003 as a public-private partnership, led by Principal Investigator Michael W. Weiner, MD. The primary goal of ADNI has been to test whether serial magnetic resonance imaging (MRI), positron emission tomography (PET), other biological markers, and clinical and neuropsychological assessment can be combined to measure the progression of mild cognitive impairment (MCI) and early Alzheimers disease (AD). For up-to-date information, see www.adni-info.org.

Both sMCI and pMCI groups have 160 patients, each with 1 PET and 1 MRI image. Early-phase images were used and the diagnosis results in late phase were used as labels.

Since ADNI is a multi-center database, imaging protocol and device varies. In general, most MRI images were in 3T.

## Image preprocessing

Three image preprocessing techniques were applied: affine registration, bias field correction and partial volume correction.

[SimpleElastix](http://simpleelastix.github.io/) was used for affine registration between PET and MRI images, and PET and PET atlas images to make sure they're aligned.

Bias field correction and partial volume correction were conducted using [PETPVE12](https://github.com/GGonEsc/petpve12) toolbox in [SPM12](https://www.fil.ion.ucl.ac.uk/spm/) to remove the noise and correct the location of PET images.

## Model structure

Since MRI and FDG-PET have their own advantage, it is a good idea to use both, but the problem is how to combine two distinctly different modalities. A common solution is to transform the original data into a higher-level representation such that we assume in the new feature space two modalities can be combined more easily. 

Here we proposed a novel structure called Dual-modal Convolutional Neural Networks (DmCNN), as shown in the figure below. We used a CNN for each modality as backend and combine the outputs of these CNNs via a Merge module, which computes the mean and variance of the two and concatenate them. In this case we try to use CNNs to learn the high-level feature representation for both. Then another CNN as frontend is used to learn and provide final prediction. 

In consideration of the difference between MRI and PET, we used different structures for two modalities. We applied Fully Convolutional Neural Network (FCNN) to PET, usually used for segmentation and thus can provide finer details while preserving global contexts for images, given the limited resolution of PET. For MRI, we used DenseNet with more focus on structural information, like lines, edges and shapes.

![DmCNN model structure](https://github.com/lizoyu/DmCNN/model_structure.png)