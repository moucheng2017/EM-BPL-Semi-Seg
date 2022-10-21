import numpy as np
import WaveFront as wf

def PTKCloseBranchesInTree(airway_tree, closing_size_mm):


    segments_to_do = []
    segments_to_do.append(airway_tree)
    segment_index = 0

    while segments_to_do:
        segment = segments_to_do[-1]
        segments_to_do.pop(-1)
        segments_to_do.extend(segment.Children)
        segment_index = segment_index + 1
        CloseSegment(segment, closing_size_mm)

    return airway_tree


def CloseSegment(segment, closing_size_mm):
    voxel_indices = segment.GetAcceptedVoxels()

    if voxel_indices:
        all_points = GetClosedIndicesForSegmentAndChildren(segment, closing_size_mm)
        new_points = list(set(all_points) ^ set(voxel_indices))
        segment.AddClosedPoints(list(set(new_points)))


def GetClosedIndicesForSegmentAndChildren(current_segment, closing_size_mm):

    segment_indices = current_segment.GetAcceptedVoxels()

    if not current_segment.Children:
        # result_points = GetClosedIndices(segment_indices, closing_size_mm, image_size)
        result_points = segment_indices

    else:
        result_points = []
        for child_segment in current_segment.Children:
            child_indices = child_segment.GetAcceptedVoxels()
            all_indices = []
            all_indices.extend(segment_indices)
            all_indices.extend(child_indices)
            new_points_all = all_indices
            new_points_children = child_indices

#            new_points_all = GetClosedIndices(all_indices, closing_size_mm, image_size)
#            new_points_children = GetClosedIndices(child_indices, closing_size_mm, image_size)
#            new_points_all =
            new_points = list(set(new_points_all) ^ set(new_points_children))
            result_points.extend(new_points)

    result_points = list(set(result_points))

    return result_points

#
# def GetClosedIndices(voxel_indices, closing_size_mm, image_size):
#     offset, segment_image = GetMinimalImageForIndices(voxel_indices)
#     border_size = 3
#     bordered_segment_image = PTKImage(segment_image)
#     bordered_segment_image.AddBorder(border_size)
#     bordered_segment_image.BinaryMorph(@imclose, closing_size_mm)
#
#     border_offset = border_size
#     bordered_image_size = bordered_segment_image.ImageSize
#     all_points = np.argwhere(bordered_segment_image > 0)
#     new_points = MimImageCoordinateUtilities.OffsetIndices(int32(all_points), -border_offset + int32(offset), int32(bordered_image_size), int32(image_size))
#
#
#     return new_points

def GetMinimalImageForIndices(indices):

    i = [item[0] for item in indices]
    j = [item[1] for item in indices]
    k = [item[2] for item in indices]

    mins = np.asarray([min(i), min(j), min(k)])
    maxs = np.asarray([max(i), max(j), max(k)])
    reduced_image_size = maxs - mins + np.asarray([3, 3, 3])
    reduced_image = np.zeros(reduced_image_size)
    offset = mins

    new_indecies = list.copy(indices)
    for i in range(0, len(indices)):
        new_indecies[i] = [new_indecies[i][0] - offset[0], new_indecies[i][1] - offset[1], new_indecies[i][2] - offset[2]]

    for i in range(0, len(indices)):
        pos = new_indecies[i]
        reduced_image[pos[0], pos[1], pos[2]] = 1

    return offset, reduced_image
