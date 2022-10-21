import nibabel as nib
import numpy as np
import WaveFront as wf
import CloseBranchesInTree as cbit
from scipy import ndimage

def PTKAirwayRegionGrowingWithExplosionControl(threshold_image, voxel_size, start_point_global, maximum_number_of_generations, explosion_multiplier):

    # PTKAirwayRegionGrowingWithExplosionControl. Segments the airways from a threshold image using a
    # region growing method.

    # Inputs:
    #     threshold_image - a lung volume stored as a PTKImage which has been
    #         thresholded for air voxels (1=air, 0=background).
    #         Note: the lung volume can be a region-of-interest, or the entire
    #         volume.
    #
    #     start_point_global - coordinate (i,j,k) of a point inside and near the top
    #         of the trachea, in global coordinates (as returned by plugin
    #         PTKTopOfTrachea)
    #
    #     maximum_number_of_generations - tree-growing will terminate for each
    #         branch when it exceeds this number of generations in that branch
    #
    #     explosion_multiplier - 7 is a typical value. An explosion is detected
    #         when the number of new voxels in a wavefront exceeds the previous
    #         minimum by a factor defined by this parameter
    #
    #
    # Outputs:
    #     results - a structure containing the following fields:
    #         airway_tree - a PTKTreeSegment object which represents the trachea.
    #             This is linked to its child segments via its Children property, and so on, so the entre tree can be
    #             accessed from this property.
    #         explosion_points - Indices of all voxels marked as explosions during the region-growing process.
    #         endpoints - Indices of final points in each branch of the airway tree
    #         start_point - the trachea location as passed into the function
    #         image_size - the image size

    # Perform the airway segmentation
    airway_tree = RegionGrowing(threshold_image, voxel_size, start_point_global, maximum_number_of_generations, explosion_multiplier)

    # Sanity checking and warn user if any branches terminated early
#    CheckSegments(airway_tree)

    # Find points which indicate explosions
    explosion_points = GetExplosionPoints(airway_tree)

    # Remove segments in which all points are marked as explosions
    airway_tree = RemoveCompletelyExplodedSegments(airway_tree)

    # Remove holes within the airway segments
    closing_size_mm = 5

    suppress_small_structures = True
    airways_mask_im = GetImageFromAirwayResults(airway_tree, threshold_image, suppress_small_structures)

    explosion_points = np.asarray(explosion_points)
    endpoints = FindEndpointsInAirwayTree(airway_tree)

    top_of_trachea = tuple(start_point_global[round(len(start_point_global)/2)])
    endpoints.append(top_of_trachea)

    airways_mask_im = ndimage.morphology.binary_closing(airways_mask_im)

    return airways_mask_im, endpoints


def RegionGrowing(threshold_image, voxel_size_mm, start_point_index, maximum_number_of_generations, explosion_multiplier):

    frontmost_points = []
    min_distance_before_bifurcating_mm = max(3, np.ceil(threshold_image.shape[2]*voxel_size_mm[2])/4)
    image_size = threshold_image.shape

    first_segment = wf.Wavefront_class()
    first_segment.Wavefront([], min_distance_before_bifurcating_mm, voxel_size_mm,
                                 maximum_number_of_generations, explosion_multiplier)

    for i in range(0, len(start_point_index)):
        pos = start_point_index[i]
        threshold_image[pos[0], pos[1], pos[2]] = 0
        threshold_image[:, :, pos[2]+1] = 0

    segments_in_progress = []
    first_segment.AddNewVoxelsAndGetNewSegments(start_point_index, image_size)
    segments_in_progress.append(first_segment)
    il = 0

    while len(segments_in_progress):
        il = il+1
        # Get the next airway segment to add voxels to
        current_segment = segments_in_progress[0]
        segments_in_progress.pop(0)

        # Fetch the front of the wavefront for this segment
        frontmost_points = current_segment.GetFrontmostWavefrontVoxels()

        # Find the neighbours of these points, which will form the next generation of points to add to the wavefront
        indices_of_new_points = GetNeighbouringPoints(frontmost_points)

        in_range = [item for item in indices_of_new_points if item[0] >= 0 and item[1] >= 0 and item[2] >= 0]
        indices_of_new_points = [item for item in in_range if item[0] < image_size[0] and item[1] < image_size[1] and
                                 item[2] < image_size[2] and threshold_image[item] == 1]

        # If there are no new candidate neighbour indices then complete the segment
        if len(indices_of_new_points) == 0:

            current_segment.CompleteThisSegment()

        else:
            for i in range(0, len(indices_of_new_points)):
                pos = indices_of_new_points[i]
                threshold_image[pos[0], pos[1], pos[2]] = 0

            # Add points to the current segment and retrieve a list of segments which require further processing -
            # this can comprise of the current segment if it is incomplete, or child segments if it has bifurcated

            next_segments = current_segment.AddNewVoxelsAndGetNewSegments(indices_of_new_points, image_size)
            segments_in_progress.extend(next_segments)

    first_segment = first_segment.CurrentBranch

    return first_segment

def  calculate_direction_vectors():
    direction_vectors = np.empty([6, 3])
    direction_vectors[0] = [-1, 0, 0]
    direction_vectors[1] = [0, -1, 0]
    direction_vectors[2] = [0, 0, -1]
    direction_vectors[3] = [1, 0, 0]
    direction_vectors[4] = [0, 1, 0]
    direction_vectors[5] = [0, 0, 1]
    return direction_vectors


def GetExplosionPoints(processed_segments):
    explosion_points = []
    segments_to_do = []
    segments_to_do.append(processed_segments)
    while segments_to_do:
        next_segment = segments_to_do[0]
        segments_to_do.pop(0)
        explosion_points.extend(next_segment.GetRejectedVoxels())
        segments_to_do.extend(next_segment.Children)

    return explosion_points

def GetNeighbouringPoints(point_indices):

    list_of_point_indices = []
    direction_vectors = calculate_direction_vectors()
    for i in range(0, len(point_indices)):
        for il in range(0, len(direction_vectors)):
            indices_to_add = list(np.asarray(point_indices[i]) + np.asarray(direction_vectors[il], dtype=np.int))
            list_of_point_indices.append(indices_to_add)

    list_of_point_indices = set(tuple(i) for i in list_of_point_indices)
    list_of_point_indices = list(list_of_point_indices)

    return list_of_point_indices


def RemoveCompletelyExplodedSegments(airway_tree):
    segments_to_do = []
    segments_to_do.append(airway_tree)
    while segments_to_do:
        next_segment = segments_to_do[0]
        segments_to_do.pop(0)

        # Remove segments which are explosions
        if len(next_segment.GetAcceptedVoxels()) == 0:
            next_segment.CutFromTree()

            # Removing an explosion may leave its parent with only one child, in
            # which case these should be merged, since they are really the same segment
            if len(next_segment.Parent.Children) == 1:
                remaining_child = next_segment.Parent.Children
                next_segment.Parent.MergeWithChild()

                # Need to be careful when merging a child branch with its parent - if the child branch is currently in
                # the segments_to_do we need to remove it and put its child branches in so they get processed correctly
                if remaining_child in segments_to_do:
                    segments_to_do.remove(remaining_child)
                    segments_to_do.extend(remaining_child.Children)

        segments_to_do.extend(next_segment.Children)

    airway_tree.RecomputeGenerations(1)

    return airway_tree


def FindEndpointsInAirwayTree(airway_tree):
    endpoints = []
    segments_to_do = []
    segments_to_do.append(airway_tree)
    while segments_to_do:
        segment = segments_to_do[0]
        segments_to_do.pop(0)
        segments_to_do.extend(segment.Children)

        if not segment.Children:
            final_voxels_in_segment = segment.GetEndpoints()
            endpoints.append(final_voxels_in_segment)

    return endpoints

def GetImageFromAirwayResults(airway_tree, template_image, suppress_small_structures):
    airways_mask_im = np.zeros_like(template_image)
    segments_to_do = []
    segments_to_do.append(airway_tree)
    while segments_to_do:
        segment = segments_to_do[0]
        segments_to_do.pop(0)
        voxels = segment.GetAllAirwayPoints()

        show_this_segment = (not suppress_small_structures) or (len(voxels) > 10) or (segment.Children)

        if show_this_segment:
            for i in range(len(voxels)):
                pos = voxels[i]
                airways_mask_im[pos[0], pos[1], pos[2]] = 1

        segments_to_do.extend(segment.Children)

    return airways_mask_im
