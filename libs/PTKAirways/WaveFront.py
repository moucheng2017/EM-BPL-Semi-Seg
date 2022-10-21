import numpy as np
import AirwayRegionGrowingWithExplosionControl as argwxp
import TreeSegment as ts
from scipy import ndimage
import matplotlib.pyplot as plt


class Wavefront_class:

    # PTKWavefront. A data structure representing a segmented airway tree
    #
    #     PTKWavefront is used as part of the airway region growing process in
    #     PTKAirwayRegionGrowingWithExplosionControl.
    #
    #     A root PTKTreeSegment is returned by PTKAirwayRegionGrowingWithExplosionControl. From this you
    #     can extract and analyse the resulting airway tree.
    #
    #     PTKTreeSegment is used in the construction and storage of
    #     the airway trees. A PTKTreeSegment stores an individual
    #     segment of the centreline tree, with references to the parent and child
    #     PTKTreeSegments, so that it is possible to reconstruct the entire
    #     tree from a single segment.


    def __init__(self):

        self.CurrentBranch = []

        # The wavefront is a thick layer of voxels which is used to detect and process bifurcations
        # in the airway tree before these voxels are added to the segement's list of pending indices
        self.WavefrontVoxelIndices = []

        # Additional voxels that were not originally part of the segment but
        # were added later after a morphological closing operation
        self.ClosedPoints = []

        self.NumberOfVoxelsSkipped = 0

        self.WavefrontSize = 1
        self.PermittedVoxelSkips = 0

        self.MinimumDistanceBeforeBifurcatingMm = 1
        self.MinimumChildDistanceBeforeBifurcatingMm = 5

        self.FirstSegmentWavefrontSizeMm = 10
        self.ChildWavefrontSizeMm = 5  # This parameter changes the most
        self.VoxelSizeMm = 1
        self.MinCoords = None
        self.MaxCoords = None

        # Generations greater than this are automatically terminated
        self.MaximumNumberOfGenerations = 10

        self.MinimumNumberOfPointsThresholdMm3 = 3

        self.ExplosionMultiplier = 7


    def Wavefront(self, segment_parent, min_distance_before_bifurcating_mm,
                  voxel_size_mm, maximum_generations, explosion_multiplier):

        self.WavefrontVoxelIndices = []

        self.MinimumDistanceBeforeBifurcatingMm = min_distance_before_bifurcating_mm
        max_voxel_size_mm = max(voxel_size_mm[0], voxel_size_mm[1], voxel_size_mm[2])
        self.VoxelSizeMm = voxel_size_mm
        self.MaximumNumberOfGenerations = maximum_generations
        self.ExplosionMultiplier = explosion_multiplier

        voxel_volume = voxel_size_mm[0]*voxel_size_mm[1]*voxel_size_mm[2]
        min_number_of_points_threshold = max(3, round(self.MinimumNumberOfPointsThresholdMm3/voxel_volume))

        if segment_parent:
            self.WavefrontSize = np.ceil(self.ChildWavefrontSizeMm/max_voxel_size_mm)
            self.CurrentBranch = ts.PTKTreeSegment()
            self.CurrentBranch.PTKTreeSegment(segment_parent, min_number_of_points_threshold, explosion_multiplier)
        else:
            self.WavefrontSize = np.ceil(self.FirstSegmentWavefrontSizeMm/max_voxel_size_mm)
            self.CurrentBranch = ts.PTKTreeSegment()
            self.CurrentBranch.PTKTreeSegment([], min_number_of_points_threshold, explosion_multiplier)

    # Returns the very front layer of voxels at the wavefront
    def GetFrontmostWavefrontVoxels(self):
        frontmost_points = self.WavefrontVoxelIndices[-1]
        return frontmost_points

    # Returns the wavefront for this segment, which includes voxels that may be separated into child segments
    def GetWavefrontVoxels(self):
        wavefront_voxels = self.ConcatenateVoxels(self.WavefrontVoxelIndices)
        return wavefront_voxels

    # Add new voxels to this segment, and returns a list of all segments that require further processing
    # (including this one, and any child segments which have been created as a result of bifurcations)
    def AddNewVoxelsAndGetNewSegments(self, indices_of_new_points, image_size):

        segments_to_do = []
        growing_branches = []
        points_by_branches = []
        s = np.ones((3, 3, 3))

        # First we move voxels at the rear of the wavefront into the PendingVoxels
        if self.WavefrontVoxelIndices:
            while len(self.WavefrontVoxelIndices) > self.WavefrontSize:
                self.MoveVoxelsFromRearOfWavefrontToPendingVoxels()

        # Next add the new points to the front of the wavefront
        self.WavefrontVoxelIndices.append(indices_of_new_points)

        # If an explosion has been detected then do not continue
        if self.CurrentBranch.MarkedExplosion:
            self.MoveAllWavefrontVoxelsToPendingVoxels()
            return []  # This segment has been terminated

        self.AdjustMaxAndMinForVoxels(indices_of_new_points)

        # Do not allow the segment to bifurcate until it is above a minimum length
        if not self.MinimumLengthPassed():
            segments_to_do.append(self)  # This segment is to continue growing
            return segments_to_do

        # Do not allow the segment to bifurcate until it is above a minimum size
        if not self.WavefrontIsMinimumSize():
            segments_to_do.append(self)  # This segment is to continue growing
            return segments_to_do

        # Determine whether to continue growing the current segment, or to split it into a new set of child segments
        # Find connected components from the wavefront (which is several voxels thick)

        [offset, reduced_image] = self.GetMinimalImageForIndices(self.GetWavefrontVoxels())
        wavefront_connected_components, number_of_components = ndimage.label(reduced_image, structure=s)


        # If there is only one component, it will be growing, so there is no need to do any further component analysis
        if number_of_components == 1:
            segments_to_do.append(self)  # This segment is to continue growing
            return segments_to_do

        # if number_of_components == 2:
        #     print('separation')

        # Iterate over the components and separate the wavefront voxels of the current segment into a new branch
        # for each growing component
        for component_number in range(1, number_of_components+1):

            # points_by_branches = []
            # Get voxel list, and offset to match those for the full image
            indices_of_component_points = np.argwhere(wavefront_connected_components == component_number)
            indices_of_component_points = indices_of_component_points + offset
            indices_of_component_points = indices_of_component_points.tolist()

            points_by_branches.append(indices_of_component_points)

            still_growing = self.IsThisComponentStillGrowing(indices_of_component_points)
            if still_growing:
                growing_branches.append(component_number)

        if len(growing_branches) < 1:
            return []

        if len(growing_branches) == 1:
            segments_to_do.append(self)  # This segment is to continue growing
            return segments_to_do

        if len(growing_branches) > 1:

            if self.MaximumNumberOfGenerations > 0:
                # If the maximum permitted number of generations is exceeded then terminate this segment. This will
                # discard any remaining wavefront voxels and mark the segment as incomplete (unless it is marked as exploded)
                if self.CurrentBranch.GenerationNumber >= self.MaximumNumberOfGenerations:
                    self.CurrentBranch.EarlyTerminateBranch()
                    return []

            for index in range(0, len(growing_branches)):
                temp = self.SpawnChildFromWavefrontVoxels(points_by_branches[growing_branches[index] - 1])
                segments_to_do.append(temp)

            # If the branch has divided, there may be some unaccepted points left over
            self.CompleteThisSegment()

        return segments_to_do

    def CompleteThisSegment(self):

        self.MoveAllWavefrontVoxelsToPendingVoxels()
        self.CurrentBranch.CompleteThisSegment()

    def MoveAllWavefrontVoxelsToPendingVoxels(self):

        while self.WavefrontVoxelIndices:
            self.MoveVoxelsFromRearOfWavefrontToPendingVoxels()

    def MoveVoxelsFromRearOfWavefrontToPendingVoxels(self):
        # The wavefront may be empty after voxels have been divided amongst child branches
        if self.WavefrontVoxelIndices[0]:
            self.CurrentBranch.AddPendingVoxels(self.WavefrontVoxelIndices[0])
        self.WavefrontVoxelIndices.pop(0)

    def AdjustMaxAndMinForVoxels(self, voxel_indices):
        x = [item[0] for item in voxel_indices]
        y = [item[1] for item in voxel_indices]
        z = [item[2] for item in voxel_indices]
        mins = [min(x), min(y), min(z)]
        maxs = [max(x), max(y), max(z)]
        if not self.MinCoords:
            self.MinCoords = mins
            self.MaxCoords = maxs
        else:
            self.MinCoords = min(mins, self.MinCoords)
            self.MaxCoords = max(maxs, self.MaxCoords)


    def WavefrontIsMinimumSize(self):
        is_minimum_size = (len(self.WavefrontVoxelIndices) >= self.WavefrontSize)
        return is_minimum_size


    def MinimumLengthPassed(self):
        lengths = np.asarray(self.MaxCoords) - np.asarray(self.MinCoords)
        lengths = np.multiply(lengths, self.VoxelSizeMm)
        max_length = np.max(lengths)

        passed_minimum_lengths = max_length >= self.MinimumDistanceBeforeBifurcatingMm

        return passed_minimum_lengths

    def SpawnChildFromWavefrontVoxels(self, voxel_indices):
        wavefront_new = list()
        voxel_indices_set = set(tuple(x) for x in voxel_indices)

        for index in range(0, len(self.WavefrontVoxelIndices)):
            wavefront_old_temp = set(tuple(x) for x in self.WavefrontVoxelIndices[index])
            wavefront_new_temp = voxel_indices_set.intersection(wavefront_old_temp)
            wavefront_new.append(list(wavefront_new_temp))
            self.WavefrontVoxelIndices[index] = list(wavefront_old_temp.difference(wavefront_new_temp))

        new_segment = Wavefront_class()
        new_segment.Wavefront(self.CurrentBranch, self.MinimumChildDistanceBeforeBifurcatingMm, self.VoxelSizeMm, self.MaximumNumberOfGenerations, self.ExplosionMultiplier)
        new_segment.WavefrontVoxelIndices = wavefront_new

        return new_segment

    def IsThisComponentStillGrowing(self, voxel_indices):
        voxel_indices_set = set(tuple(x) for x in voxel_indices)
        temp = set(tuple(x) for x in self.WavefrontVoxelIndices[-1])
        wavefront_voxels_end = voxel_indices_set.intersection(temp)
        if wavefront_voxels_end:
            still_growing = True
        else:
            still_growing = False

        return still_growing


    def ConcatenateVoxels(self, voxels):
        concatenated_voxels = []
        number_layers = len(voxels)
        for index in range(0, number_layers):
            next_voxels = voxels[index]
            if next_voxels:
                concatenated_voxels.extend(next_voxels)
        return concatenated_voxels


    def GetMinimalImageForIndices(self, indices):

        i = [item[0] for item in indices]
        j = [item[1] for item in indices]
        k = [item[2] for item in indices]

        # voxel_coordinates = [i' j' k']
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

