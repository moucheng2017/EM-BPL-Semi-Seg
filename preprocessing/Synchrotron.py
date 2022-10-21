import os
import SimpleITK as sitk
os.chdir('E:\Dropbox (UCL)\PPFE\Exported data\GLE689_top_seg')
import numpy as np
# load and display an image with Matplotlib
# from matplotlib import image
from matplotlib import pyplot as plt
from PIL import Image

# change here for image folder:
image_folder = ''
# read all tiff files in the folder
all_images = sorted(glob.glob(os.path.join(image_folder, '*.tiff')))
print(all_images)

GLE_labels_vol = np.zeros((4096, 4096, 512))

for i in all_images:
    image = Image.open(i)
    image = np.array(image).squeeze()
    GLE_labels_vol[:, :, i] = image


# summarize shape of the pixel array
print(GLE_labels_vol.dtype)
print(GLE_labels_vol.shape)
# display the array of pixels as an image
plt.imshow(GLE_labels_vol[:, :, 0])
plt.show()