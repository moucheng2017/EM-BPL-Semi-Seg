class PTKTree:

    def __init__(self):
        self.Parent = []    # Parent PTKTree
        self.Children = []  # Child PTKTree

    def PTKTree(self, parent):
        # Create an empty PTKTree segment with an optional parent segment

        self.Parent = parent

    def GetRoot(self):
        # Get the topmost branch from the tree containing this branch

        root = PTKTree()

        while len(root.Parent) > 0:
            root = root.Parent

        return root

    def CutFromTree(self):
        # Remove this branch from the tree

        if self.Parent and (self in self.Parent.Children):
            self.Parent.Children.remove(self)


    def CutAndSplice(self):
        # Remove this branch from the tree, connecting its children to its parent branch

        if self.Parent:
            self.Parent.Children = [list(set(self.Parent.Children) ^ set(self)), self.Children]


    def RemoveChildren(self):
        # Remove all child branches from this branch

        self.Children = []


    def PruneDescendants(self, num_generations_to_keep):
        # Keeps a given number of generations, and remove descendants of those

        if num_generations_to_keep <= 0:
            self.RemoveChildren()
        else:
            for branch in self.Children:
                branch.PruneDescendants(num_generations_to_keep - 1)


    def CountBranches(self):
        # Return the number of branches in this tree, from this branch downwards
        branches_to_do = []
        number_of_branches = 0
        branches_to_do.append(self)

        while branches_to_do:
            branch = branches_to_do[-1]
            branches_to_do.pop(-1)
            branches_to_do.append(branch.Children)
            number_of_branches = number_of_branches + 1

        return number_of_branches

    def ContainsBranch(self, branch):
        # Return true if this subtree contains the branch

        segments_to_do = self
        while segments_to_do:
            segment = segments_to_do[-1]
            if (segment == branch):
                contains_branch = True
 #               return

            segments_to_do.pop(-1)
            children = segment.Children
            segments_to_do .append(children)

        contains_branch = False

        return contains_branch

    def CountTerminalBranches(self):
        # Return the number of branches in this tree, from this branch downwards

        number_of_branches = 0
        branches_to_do = self
        while branches_to_do:
            branch = branches_to_do[-1]
            branches_to_do.pop(-1)
            branches_to_do.append(branch.Children)
            if not branch.Children:
                number_of_branches = number_of_branches + 1

        return number_of_branches

    def GetMinimumTerminalGeneration(self):
        # Returns the number of branches in this tree, from this branch downwards

        minimum_generation = 99
        branches_to_do = self
        while branches_to_do:
            branch = branches_to_do[-1]
            branches_to_do.pop(-1)
            branches_to_do.extend(branch.Children)
            if not branch.Children:
                minimum_generation = min(minimum_generation, branch.GenerationNumber)
        return minimum_generation


    def GetBranchesAsList(self):
        # Return all the branches as a set, from this branch onwards
        branches_list = self
        branches_to_do = self.Children

        while branches_to_do:
            branch = branches_to_do[-1]
            branches_to_do.pop(-1)
            branches_to_do.append(branch.Children)
            branches_list.append(branch)
        return branches_list

    def GetBranchesAsListByGeneration(self):
        # Returns all the branches as a set, with this branch first, then its children, then its grandchildren, and so on.

        current_generation = self
        branches_list = current_generation.empty()

        while current_generation:
            next_generation = current_generation.empty()
            for branch in current_generation:
                branches_list.append(branch)
                next_generation.append(branch.Children)

            current_generation = next_generation

        return branches_list

    def GetBranchesAsListUsingRecursion(self):
        # Returns all the branches as a set, from this branch onwards. This is similar to GetBranchesAsList, but the
        # branches are assembled in a different order. This is simply to match the output produced by other code

        branches_list = self
        for child in self.Children:
            branches_list.append(child.GetBranchesAsListUsingRecursion)

        return branches_list

    def AddChild(self, child):
        self.Children.append(child)


