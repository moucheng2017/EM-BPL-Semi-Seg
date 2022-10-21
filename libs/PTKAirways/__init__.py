import os, sys
 # cheat to add current cwd to python path, avoids relative import issues
sys.path.append(os.path.join(os.path.dirname(__file__)))

import numpy as np
from libs.PTKAirways import TopOfTrachea as tot, AirwayRegionGrowingWithExplosionControl as argwxp
from scipy. ndimage import gaussian_filter


def RunPTKAirways(image, voxel_size, smooth=True, air_threshold=-775, maximum_number_of_generations=10, explosion_multiplier=7):
    # get air mask
    if smooth is True:
        image = gaussian_filter(image, sigma=0.5)
    airmask = np.zeros_like(image)
    airmask[image <= air_threshold] = 1

    # get trachea seed
    _, top_of_trachea = tot.PTKFindTopOfTrachea(airmask, voxel_size)
    # run PTK's region growing algorithm
    airway_mask, _ = argwxp.PTKAirwayRegionGrowingWithExplosionControl(airmask, voxel_size, top_of_trachea, maximum_number_of_generations, explosion_multiplier)


    return airway_mask
