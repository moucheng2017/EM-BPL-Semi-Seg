import numpy as np
from scipy.ndimage import label, generate_binary_structure
from skimage import measure
from scipy import ndimage
import raster_geometry as rsg



def PTKFindTopOfTrachea(threshold_image, voxel_size):

    # set top and bottom axial slices to 0.
    threshold_image[:, :, 0:2] = 0
    threshold_image[:, :, -3:-1] = 0

    image_size = np.asarray(threshold_image.shape)

    midpoint_roi = np.round(image_size / 2)

    # Compute the number of voxels over which we will search for the trachea - typically this will be about half
    # of the image size in x, y and a third of the image length in the z direction
    # PTK = [200, 150, 130];
    search_length_mm = np.array([100, 200, 100])
    search_length_voxels = np.round(np.divide(search_length_mm, voxel_size))

    # Compute the starting and ending coordinates over which to search in the x and y directions
    startpoint = midpoint_roi - np.round(search_length_voxels / 2)
    endpoint = midpoint_roi + np.round(search_length_voxels / 2)
    startpoint = np.maximum(np.ones_like(startpoint), startpoint)

    # Compute the start and end coordinates in the z direction
    startpoint[2] = image_size[2] - search_length_voxels[2]
    endpoint[2] = image_size[2]

    endpoint = np.minimum(image_size, endpoint)
    startpoint = startpoint.astype('int')
    endpoint = endpoint.astype('int')

    # Crop the image
    partial_image = threshold_image[startpoint[0]: endpoint[0], startpoint[1]: endpoint[1], startpoint[2]: endpoint[2]]

    # Iterate through 2 - slice thick segments and remove components that are too wide or which touch the edges.
    # The first pass helps to disconnect the trachea from the rest of the lung, so that less of the
    # trachea is removed in the second pass.

    partial_image2 = ReduceImageToCentralComponents(partial_image, int(np.ceil(5 / voxel_size[2])), voxel_size)
    partial_image2 = ReduceImageToCentralComponents(partial_image2, int(np.ceil(1.3 / voxel_size[2])), voxel_size)

    # # Reduce the image to the main component
    trachea_voxels = FindLargestComponent(partial_image2)

    # Find coordinates of highest point in the resulting image
    relative_top_of_trachea = FindHighestPoint(trachea_voxels, voxel_size)
    top_of_trachea = relative_top_of_trachea + startpoint - [1, 1, 1]

    trachea_out = np.zeros_like(threshold_image)

    trachea_out[startpoint[0]: endpoint[0], startpoint[1]: endpoint[1], startpoint[2]: endpoint[2]] = trachea_voxels

    top_points = np.argwhere(trachea_out[:, :, top_of_trachea[2]] == 1)
    top_of_trachea = list(top_of_trachea)
    top_points = list(top_points)
    for i in range(0, len(top_points)):
        top_points[i] = list(top_points[i])
        top_points[i].append(top_of_trachea[2])

    return trachea_out, list(top_points)

def FindHighestPoint(component, voxel_size):

    # Find k - coordinate of highest point in the image component
    temp = np.where(component > 0)
    k_highest = temp[2].max()

    # Get a thick slice starting from this coordinate

    slice_thickness_mm = 3
    voxel_thickness = int(np.ceil(slice_thickness_mm / voxel_size[2]))
    thick_slice = component[:, :, k_highest: k_highest + voxel_thickness]
    voxel_indices = np.where(thick_slice > 0)

    # Look for a central point at the top
    centrepoint = [(int(voxel_indices[0].mean()), int(voxel_indices[1].mean()), 0)]

    # Get coordinates of all the points
    all_points = np.argwhere(thick_slice > 0)

    # Find closest point in the component to this point
    # Return point in [i, j, k] notation
    [i, j, k] = min(all_points, key=lambda x: (abs(x[0]-centrepoint[0][0]) + abs(x[1]-centrepoint[0][1]) + abs(x[2]-centrepoint[0][2])))

    relative_top_of_trachea = [i, j, k + k_highest - 1]

    return relative_top_of_trachea


def ReduceImageToCentralComponents(image_to_reduce, slices_per_step, voxel_size):
    result = np.zeros_like(image_to_reduce)
    disc_filling = np.array(rsg.sphere(4, 1)).astype(np.int16)

    # for images with thick slices, we need to choose the minimum voxel size, not the maximum
    max_xy_voxel_size = min(voxel_size[0], voxel_size[1])

    # Compute a maximum for the trachea diameter - we will filter out structures wider than this
    max_trachea_diameter_mm = 30
    max_trachea_diameter = max_trachea_diameter_mm / max_xy_voxel_size  # typically about 80 voxels

    # Compute an additional factor for the permitted diameter to take into account that the trachea may not be vertical
    # when computing over multiple slices
    vertical_height_mm = slices_per_step * voxel_size[2]
    permitted_horizontal_trachea_movement_mm = vertical_height_mm
    permitted_horizontal_trachea_movement_voxels = permitted_horizontal_trachea_movement_mm / max_xy_voxel_size

    # The trachea may be at an angle so we take into account movement between slices by increasing the maximum
    # permitted diameter by the factor computed above
    max_trachea_diameter = max_trachea_diameter + permitted_horizontal_trachea_movement_voxels

    # We add a border in the x and y(horizontal) directions - we use this to remove components which touch these borders
    border_slice = np.zeros((np.size(image_to_reduce, 0), np.size(image_to_reduce, 1), slices_per_step))
    border_slice[0, :, :] = 2
    border_slice[-1, :, :] = 2
    border_slice[:, 0, :] = 2
    border_slice[:, -1, :] = 2
    num_slices = np.size(image_to_reduce, 2)
    num_slices = int(slices_per_step * np.floor(num_slices / slices_per_step))

    # Iterate through all slices
    for k_index in range(0, num_slices, slices_per_step):
        k_max = k_index + slices_per_step - 1
        slice_im = image_to_reduce[:, :, k_index: k_max + 1]
        slice_im = slice_im + border_slice


        slice_im[slice_im > 1] = 1

        # plt.imshow(slice_im[:, :, 0])
        # plt.show()

        s = np.array(rsg.sphere(3, 1)).astype(np.int16)
        connected_components, num_features = label(slice_im, structure=s)

        # Iterate through all components
        for component_index in range(0, num_features):
            locations = np.where(connected_components == component_index)

            # bounding_box = stats[component_index].bbox
            width = [max(locations[0]) - min(locations[0]), max(locations[1]) - min(locations[1])]

            # Remove components greater than a certain size in the x and y dimensions,
            # and remove and components which connect to the edge
            touches_edge = check_if_touches_edge(locations, slice_im)

            if (len(locations[0]) < 20) or (width[0] > max_trachea_diameter) or \
                    (width[1] > max_trachea_diameter) or touches_edge:

                slice_im[connected_components == component_index] = 0

        slice_im = ndimage.binary_fill_holes(slice_im, structure=disc_filling)
        result[:, :, k_index: k_max + 1] = slice_im
    # Ensure border is fully removed when returning result.
    result[0, :, :] = 0
    result[-1, :, :] = 0
    result[:, 0, :] = 0
    result[:, -1, :] = 0

    return result


def check_if_touches_edge(locations, slice_im):
    (size_x, size_y, size_z) = np.shape(slice_im)
    if size_x-1 in locations[0] or 0 in locations[0] or size_y-1 in locations[1] or 0 in locations[1]:
        touches_edge = True
    else:
        touches_edge = False

    return touches_edge


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
