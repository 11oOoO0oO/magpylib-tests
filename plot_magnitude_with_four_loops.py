import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy
from scipy.spatial.transform import Rotation as R

from plot_field import plot_field

def plot_magnitude_with_four_loops(*lists, loop=0):

    assert all(isinstance(lst, list) for lst in lists), "All input arguments must be lists."
    assert all(len(lst) == 5 for lst in lists), "All input lists must be formatted as [x_coordinate, y_coordinate, z_coordinate, loop_current, loop_diameter]."
    assert len(lists) == 4, "There must be exactly four loops."
    assert [lst[0] == 0 for lst in lists].count(True) == 4, "All loops must have x-coordinates of 0."
    assert [lst[1] == 5 or lst[1] == -5 for lst in lists].count(True) == 2, "There must be exactly two loops with y-coordinates of 5 and -5."
    assert [lst[2] == 5 or lst[2] == -5 for lst in lists].count(True) == 2, "There must be exactly two loops with z-coordinates of 5 and -5."

    Y, Z = np.mgrid[-5:5:256j, -5:5:256j]
    grid = np.stack([np.zeros((256, 256)), Y, Z], axis=2)

    def generator(lists, grid):

        # Compute B- and H-fields of individual source objects and map to grid

        for lst in lists:

            if lst[1] == 5 or lst[1] == -5:
                loop = magpy.current.Loop(
                    current=lst[3], 
                    diameter=lst[4],
                    position=lst[:3],
                    orientation=R.from_euler("x", 90, degrees=True)
                    )
            else:
                loop = magpy.current.Loop(
                    current=lst[3], 
                    diameter=lst[4], 
                    position=lst[:3],
                    )
            
            B = loop.getB(grid)
            H = loop.getH(grid)

            B_magnitudes = np.linalg.norm(B, axis=2)
            H_magnitudes = np.linalg.norm(H, axis=2)

            yield B_magnitudes, H_magnitudes
            yield loop
        
    gen = generator(lists, grid)
    results = []
    coll = magpy.Collection()

    for _ in range(4):
        results.append(next(gen))
        coll.add(next(gen))

    B_res, H_res = results[loop]  # Get the last yielded value (based on loop count)

    # print the results
    print("\nHere are the B magnitudes for the loop selected:\n\n" + str(B_res) + "\n\nand here are the H magnitudes for loop selected:\n\n" + str(H_res) + "\n")

    B = coll.getB(grid)
    H = coll.getH(grid)

    B_magnitudes = np.linalg.norm(B, axis=2)
    H_magnitudes = np.linalg.norm(H, axis=2)

    print("\nHere are the B magnitudes for the loop collection:\n\n" + str(B_magnitudes) + "\n\nand here are the H magnitudes for the loop collection:\n\n" + str(H_magnitudes) + "\n")

    plot_field(coll)

#Test
plot_magnitude_with_four_loops([0, 0, 5, 1, 4], [0, 0, -5, 1, 4], [0, 5, 0, 1, 4], [0, -5, 0, 1, 4], loop=1)