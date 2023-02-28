import numpy as np
import pydicom
import os
import glob
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import nibabel as nib
# import itk as itk
# import dicom2nifti
from skimage import measure, morphology, segmentation


# Load the scans in given folder path
def load_nifti(path):
    # for nifti
    slices = nib.load(path)
    return slices

def load_scan(path):
    # for DICOM
    slices = [pydicom.dcmread(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices

def get_pixels_hu(scans):
    image = np.stack(scans.get_fdata())
    #     image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    # Convert to Hounsfield units (HU)
    #     intercept = scans[0].RescaleIntercept
    #     slope = scans[0].RescaleSlope
    # new:
    slope = scans.dataobj.slope
    intercept = scans.dataobj.inter

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def generate_markers(image):
    # Creation of the internal Marker
    # print(image)
    marker_internal = image < -400
    marker_internal = segmentation.clear_border(marker_internal)
    marker_internal_labels = measure.label(marker_internal)
    areas = [r.area for r in measure.regionprops(marker_internal_labels)]
    areas.sort()
    if len(areas) > 2:
        for region in measure.regionprops(marker_internal_labels):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    marker_internal_labels[coordinates[0], coordinates[1]] = 0
    marker_internal = marker_internal_labels > 0
    # Creation of the external Marker
    external_a = ndimage.binary_dilation(marker_internal, iterations=20)
    external_b = ndimage.binary_dilation(marker_internal, iterations=55)
    marker_external = external_b ^ external_a
    # Creation of the Watershed Marker matrix
    # print(np.shape(image))
    h, w = np.shape(image)
    marker_watershed = np.zeros((h, w))
    marker_watershed += marker_internal * 255
    marker_watershed += marker_external * 128
    return marker_internal, marker_external, marker_watershed

def seperate_lungs(image):
    # Creation of the markers as shown above:
    marker_internal, marker_external, marker_watershed = generate_markers(image)

    # Creation of the Sobel-Gradient
    sobel_filtered_dx = ndimage.sobel(image, 1)
    sobel_filtered_dy = ndimage.sobel(image, 0)
    sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
    sobel_gradient *= 255.0 / np.max(sobel_gradient)

    # Watershed algorithm
    watershed = segmentation.watershed(sobel_gradient, marker_watershed)

    # Reducing the image created by the Watershed algorithm to its outline
    outline = ndimage.morphological_gradient(watershed, size=(1, 1))
    outline = outline.astype(bool)

    # Performing Black-Tophat Morphology for reinclusion
    # Creation of the disk-kernel and increasing its size a bit
    blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0, 0]]
    blackhat_struct = ndimage.iterate_structure(blackhat_struct, 8)
    # Perform the Black-Hat
    outline += ndimage.black_tophat(outline, structure=blackhat_struct)
    # Use the internal marker and the Outline that was just created to generate the lungfilter
    lungfilter = np.bitwise_or(marker_internal, outline)
    # Close holes in the lungfilter
    # fill_holes is not used here, since in some slices the heart would be reincluded by accident
    lungfilter = ndimage.binary_closing(lungfilter, structure=np.ones((5, 5)), iterations=5)
    # # Apply the lungfilter (note the filtered areas being assigned -2000 HU)
    segmented = np.where(lungfilter == 1, image, -2000 * np.ones((512, 512)))
    return lungfilter, segmented

def LungMask(path_ct):
    # temppath = path + '/' + '*vessel.nii*'
    # path_vessel = glob.glob(temppath)
    # path_vessel = path_vessel[0]
    # path_ct = path_vessel[0:len(path_vessel)-11]
    # path_ct = path_ct + '.nii'
    # filename, extension = os.path.splitext(path_vessel[0])
    # path_ct = filename.split("_")
    # path_ct = path_ct[0] + '_' + path_ct[1] + extension
    filename1, extension = os.path.splitext(path_ct)
    ct = load_nifti(path_ct)
    slope = ct.dataobj.slope
    intercept = ct.dataobj.inter
    # vessels = load_nifti(path_vessel)
    ct_hu = get_pixels_hu(ct)
    # vessels_hu = get_pixels_hu(vessels)
    # change here to select starting and end points
    start_slice = 0
    end_slice = np.shape(ct_hu)[2]
    for slice_in in range(start_slice, end_slice):
        # axial plane:
        ct_slice_axial = ct_hu[:, :, slice_in]
        # vessel_slice_axial = vessels_hu[:, :, slice_in]
        lungfilter_axial, _ = seperate_lungs(ct_slice_axial)
        new_ct_axial = lungfilter_axial * ct_slice_axial
        if (slice_in == start_slice):
            ct_masked = np.array(new_ct_axial)
            # selected_ct = np.array(ct_slice_axial)
            lung_mask = np.array(lungfilter_axial)
        else:
            ct_masked = np.dstack((ct_masked, new_ct_axial))
            # selected_ct = np.dstack((selected_ct, ct_slice_axial))
            lung_mask = np.dstack((lung_mask, lungfilter_axial))
        print('slice ' + str(slice_in) + ' is done')
        print('\n')

    cts_masked = (ct_masked.astype(np.float64) - np.int16(intercept)) / slope
    # lung_mask = (lung_mask.astype(np.float64) - np.int16(intercept)) / slope
    # selected_ct = (selected_ct.astype(np.float64) - np.int16(intercept)) / slope
    store_path = filename1 + '_ct_masked.nii'
    store_path_lungfilter = filename1 + '_lung_filter.nii'
    nib.save(nib.Nifti1Image(lung_mask, ct.affine, ct.header), store_path_lungfilter)
    nib.save(nib.Nifti1Image(cts_masked, ct.affine, ct.header), store_path)
    print('case ' + filename1 + 'lung masked vessel is done')
    print('\n')
    print('\n')
    print('\n')
    return nib.load(store_path)

def run_all_cases(path, tag):
    folders = os.listdir(path)
    for f in folders:
        full_f = os.path.join(path, f)
        # LungMaskVessel(full_f, tag)
        # print('case is done...')
        fff = os.listdir(full_f)
        for ffff in fff:
            full_ffff = os.path.join(full_f, ffff)
            LungMask(full_ffff, tag)
            print('case is done...')
            print('\n')


if __name__ == '__main__':
    data_path = '/home/moucheng/projects_data/Pulmonary_data/airway/Mixed/inference/imgs/Pat25b.nii.gz'
    # ct = load_nifti(data_path)
    # slope = ct.dataobj.slope
    # intercept = ct.dataobj.inter
    # ct_hu = get_pixels_hu(ct)
    # _, segmented = seperate_lungs(ct_hu)
    LungMask(data_path)
    # path = '/home/moucheng/projects data/Pulmonary data/sarcoid patients CT/HRCT nii'
    # # path = '/home/moucheng/projects data/Pulmonary data/IPF patients/Contrast Enhanced CT/contrast enhanced cases'
    # print('calculation started..')
    # # # create all lung vessels first
    # # Parallel(n_jobs=2, prefer="threads")(run_all_cases_vessels(path))
    # # create all lung masks now and apply them on the lung vessels
    # Parallel(n_jobs=2, prefer="threads")(run_all_cases(path, 'new'))
    # print('End')