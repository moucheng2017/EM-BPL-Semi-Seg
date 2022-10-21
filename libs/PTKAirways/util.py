import numpy as np
from scipy.ndimage import label

def FindLargestComponent(mask):
    result = np.zeros_like(mask)
    s = np.ones((3, 3, 3))
    connected_components, num_features = label(mask, structure=s)
    max_number_voxels = 0
    trachea_value = 0

    for component_value in range(1, num_features):
        pixels = np.argwhere(connected_components == component_value)

        if len(pixels) > max_number_voxels:
            max_number_voxels = len(pixels)
            trachea_value = component_value

    result[connected_components == trachea_value] = 1

    return result
