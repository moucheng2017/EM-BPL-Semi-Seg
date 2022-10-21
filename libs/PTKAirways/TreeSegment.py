import numpy as np
import Tree as Tree


class PTKTreeSegment(Tree.PTKTree):

    def __init__(self):

        Tree.PTKTree.__init__(self)

        self.Colour = []        # A colourmap index allocated to this branch
        # Generation of this segment, starting at 1
        self.GenerationNumber = None

        # Indicates if this branch was terminated due to the generation number being too high, which may indicate leakage
        self.ExceededMaximumNumberOfGenerations = False

        self.MarkedExplosion = False

        # List of accepted (i.e. non-exploded) voxels
        self.AcceptedVoxelIndices = []

        # List of rejected (exploded) voxels
        self.RejectedVoxelIndices = []

        # Indices are allocated to the segment from the back of the wavefront They are pending until the explosion control
        # heuristic has determined they are 'Accepted' or 'Exploded'.
        self.PendingVoxelIndices = []

        # Additional voxels that were not originally part of the segment but were added later after a morphological closing operation
        self.ClosedPoints = []

        self.PreviousMinimumVoxels = None
        self.NumberOfVoxelsSkipped = 0
        self.LastNumberOfVoxels = 0

        self.PermittedVoxelSkips = 0
        self.IsFirstSegment = True

        # We never use less than this value at each step when computing the minimum wavefront size over the image
        self.MinimumNumberOfPointsThreshold = None

        self.ExplosionMultiplier = None

    def PTKTreeSegment(self, parent, min_number_of_points_threshold, explosion_multiplier):

        self.Parent = parent
        self.MarkedExplosion = False
        self.PendingVoxelIndices = []
        self.AcceptedVoxelIndices = []
        self.RejectedVoxelIndices = []

        self.ExplosionMultiplier = explosion_multiplier
        self.MinimumNumberOfPointsThreshold = min_number_of_points_threshold

        if parent:
            parent.AddChild(self)
            self.PreviousMinimumVoxels = parent.PreviousMinimumVoxels
            self.LastNumberOfVoxels = self.PreviousMinimumVoxels
            self.IsFirstSegment = False
            self.GenerationNumber = parent.GenerationNumber + 1
        else:
            self.PreviousMinimumVoxels = 1000
            self.LastNumberOfVoxels = self.PreviousMinimumVoxels
            self.IsFirstSegment = True
            self.GenerationNumber = 1

    # Returns a list of voxels which have been accepted as not explosions.
    def GetAcceptedVoxels(self):
        accepted_voxels = self.ConcatenateVoxels(self.AcceptedVoxelIndices)
        return accepted_voxels

    # Returns the wavefront for this segment, which includes voxels that may be separated into child segments
    def GetRejectedVoxels(self):
        rejected_voxels = self.ConcatenateVoxels(self.RejectedVoxelIndices)
        return rejected_voxels

    def GetEndpoints(self):
        endpoints = self.AcceptedVoxelIndices[-1]
        return endpoints

    # Returns all accepted region-growing points, plus those added from the airway closing operation
    def GetAllAirwayPoints(self):
        all_points = []
        all_points.extend(self.GetAcceptedVoxels())
        all_points.extend(self.ClosedPoints)
        return all_points

    # Points which are added later to close gaps in the airway tree
    def AddClosedPoints(self, new_points):
        self.ClosedPoints.extend(new_points)

    def RecomputeGenerations(self, new_generation_number):
        self.GenerationNumber = new_generation_number
        children = self.Children
        for child in children:
            child.RecomputeGenerations(new_generation_number + 1)

    # Returns the number of branches in this tree, from this branch and excluding generations above the max_generation_number
    def CountBranchesUpToGeneration(self, max_generation_number):
        number_of_branches = 0
        if self.GenerationNumber > max_generation_number:
            print('')
            return None

        else:
            branches_to_do = self
            while branches_to_do:
                branch = branches_to_do[0]
                branches_to_do.pop(0)
                if branch.GenerationNumber <= max_generation_number:
                    branches_to_do.append(branch.Children)
                    number_of_branches = number_of_branches + 1
        return number_of_branches

    def AddColourValues(self, new_colour_value):
        self.Colour = new_colour_value
        children = self.Children
        for child in children:
            new_colour_value = new_colour_value + 1
            if (new_colour_value % 7) == 0:
                new_colour_value = new_colour_value + 1

            child.AddColourValues(new_colour_value)

    def AddPendingVoxels(self, indices_of_new_points):

        if self.PendingVoxelIndices:
            self.LastNumberOfVoxels = max(1, round(len(indices_of_new_points)/2))

        self.PendingVoxelIndices.extend(indices_of_new_points)

        if self.MarkedExplosion:
            self.RejectAllPendingVoxelIndices()
            return None

        number_of_points = len(indices_of_new_points)

        if (number_of_points < self.PreviousMinimumVoxels) and (not self.IsFirstSegment):
            self.PreviousMinimumVoxels = max(number_of_points, self.MinimumNumberOfPointsThreshold)

        if number_of_points < self.ExplosionMultiplier*self.PreviousMinimumVoxels:
            self.NumberOfVoxelsSkipped = 0
        else:
            self.NumberOfVoxelsSkipped = self.NumberOfVoxelsSkipped + 1

        # Keep track of the point at which an explosion starts to occur
        if number_of_points <= self.LastNumberOfVoxels:
            self.LastNumberOfVoxels = number_of_points
            self.AcceptAllPendingVoxelIndices()

        # Explosion control: we allow a certain number of consecutive points to exceed the expansion limit. Once exceeded,
        # this segment is not permitted to expand further, and it is also marked so that it can be deleted later.
        if self.NumberOfVoxelsSkipped > self.PermittedVoxelSkips:
            self.MarkedExplosion = True
            self.RejectAllPendingVoxelIndices()

    def EarlyTerminateBranch(self):
        self.CompleteThisSegment()
        if not self.MarkedExplosion:
            self.ExceededMaximumNumberOfGenerations = True

    def CompleteThisSegment(self):
        if self.MarkedExplosion:
            self.RejectAllPendingVoxelIndices()
        else:
            self.AcceptAllPendingVoxelIndices()

    def MergeWithChild(self):

        if len(self.Children) != 1:
            return

        child = self.Children[0]
        self.Children.pop(0)

        grandchildren = child.Children
        self.Children.extend(grandchildren)

        for i in range(0, len(self.Children)):
            self.Children[i].Parent = self
            self.Children[i].RecomputeGenerations(self.GenerationNumber + 1)

        while child.AcceptedVoxelIndices:
            self.AcceptedVoxelIndices.append(child.AcceptedVoxelIndices[0])
            child.AcceptedVoxelIndices.pop(0)

        while child.RejectedVoxelIndices:
            self.RejectedVoxelIndices.append(child.RejectedVoxelIndices[0])
            child.RejectedVoxelIndices.pop(0)

    def PruneAcceptedVoxels(self):
        pruned_voxels = []
        for index in range(0, len(self.AcceptedVoxelIndices)):
            next_voxel_set = self.AcceptedVoxelIndices[index]
            size_voxel_set = np.size(next_voxel_set)
            if (index == 1) or (size_voxel_set <= last_size):
                pruned_voxels[index] = next_voxel_set
                last_size = size_voxel_set
            else:
                self.AcceptedVoxelIndices = pruned_voxels
                return None

    def AcceptAllPendingVoxelIndices(self):
        while self.PendingVoxelIndices:
            self.AcceptedVoxelIndices.append(self.PendingVoxelIndices[0])
            self.PendingVoxelIndices.pop(0)

    def RejectAllPendingVoxelIndices(self):
        while self.PendingVoxelIndices:
            self.RejectedVoxelIndices.append(self.PendingVoxelIndices[0])
            self.PendingVoxelIndices.pop(0)

    def ConcatenateVoxels(self, voxels):
        concatenated_voxels = []
        number_layers = len(voxels)
        for index in range(0, number_layers):
            next_voxels = voxels[index]
            if next_voxels:
                concatenated_voxels.append(next_voxels)

        return concatenated_voxels

