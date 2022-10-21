import numpy as np



def PTKIsSimplePoint(image, neighbours_logical_6, neighbours_logical_26):
    # PTKIsSimplePoint. Determines if a point in a 3D binary image is a simple point.
    #
    #     A point is simple if removing it does not change the local
    #     connectivity of the surrounding points.
    #
    #     A faster implementation of this function can be found in
    #     PTKFastIsSimplePoint, which uses mex.
    #
    #     Based on algirithm by G Malandain, G Bertrand, 1992
    
    is_simple = GetNumberOfConnectedComponents(image, 26, neighbours_logical_6, neighbours_logical_26) and \
                GetNumberOfConnectedComponents(1-image, 6, neighbours_logical_6, neighbours_logical_26)

    return is_simple


def GetNumberOfConnectedComponents(image, n, neighbours_logical_6, neighbours_logical_26):
    
    if n == 6:
        n6s = [[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [1, 0, 1], [0, 1, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]
        n18s = [[[0, 1, 0], [1, 1, 1], [0, 1, 0]], [[1, 1, 1], [1, 0, 1], [1, 1, 1]], [[0, 1, 0], [1, 1, 1], [0, 1, 0]]]
        points_to_connect = n6s * image
        points_that_can_be_visited = n18s * image
        neighbours_logical = neighbours_logical_6
    else:
        n26s = np.ones((3, 3, 3), dtype=bool)
        n26s[1, 1, 1] = 0
        points_to_connect = n26s * image
        points_that_can_be_visited = n26s * image
        neighbours_logical = neighbours_logical_26

    temp = np.argwhere(points_to_connect)

    if len(temp) > 0:
        index_of_first_point_to_connect = temp[0]
    # If there are no image points connected to the centre, then this is not a simple point since removing it will
    # either create a hole or remove an isolated point
    else:
        return False

    points_that_can_be_visited[tuple(index_of_first_point_to_connect)] = False
    points_to_connect[tuple(index_of_first_point_to_connect)] = False
    
    next_points = np.zeros((3, 3, 3), dtype=bool)
    next_points[tuple(index_of_first_point_to_connect)] = True

    while len(np.argwhere(next_points > 0)):
        temp = np.transpose(next_points, (0, 2, 1)).flatten()
        logical_neighbours = np.max(neighbours_logical * temp, axis=1)
        temp = logical_neighbours * np.transpose(points_that_can_be_visited, (0, 2, 1)).flatten()
        next_points = np.transpose(temp.reshape((3, 3, 3)), (0, 2, 1))
        points_that_can_be_visited[np.transpose(logical_neighbours.reshape((3, 3, 3), order='C'), (0, 2, 1)) >0] = False

    is_n_cc_1 = not any(points_to_connect.flatten() * points_that_can_be_visited.flatten())
    
    return is_n_cc_1

