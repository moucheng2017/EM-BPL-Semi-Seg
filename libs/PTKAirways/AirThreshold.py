def GetAirThresholdImage(image,lower_air_limit=-775):
    smooth_image = ndimage.gaussian_filter(image, sigma=0.5)
    lung_image_im[smooth_image > lower_air_limit] = 0
    # pad top and bottom
    lung_image_im[:, :, 0:2] = 0
    lung_image_im[:, :, -3:-1] = 0
    lung_image_im[lung_image_im <= threshold] = 1
    lung_image_header = lung_image_nii.header
    voxel_size = lung_image_header['pixdim'][1:4]
