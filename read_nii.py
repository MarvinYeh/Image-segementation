import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

data_dir = os.path.join(os.getcwd(),'Brats17_2013_2_1')

example = nib.load(os.path.join(data_dir,'Brats17_2013_2_1_t1ce.nii.gz'))

print(example.shape)
img= example.get_data()

for n in range(0,img.shape[-1]):
    plt.imshow(img[...,n])
    plt.set_cmap('gray')
    plt.pause(.0001)