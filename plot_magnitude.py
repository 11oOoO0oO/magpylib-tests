import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy
from scipy.spatial.transform import Rotation as R

from plot_field import plot_field

def plot_magnitude(*sources, loop=0):

    def generator(sources):

        Y, Z = np.mgrid[-5:5:256j, -5:5:256j]
        grid = np.stack([np.zeros((256, 256)), Y, Z], axis=2)

        # Compute B- and H-fields of individual source objects and map to grid
        for source in sources:
            
            B = source.getB(grid)
            H = source.getH(grid)

            B_magnitudes = np.linalg.norm(B, axis=2)
            H_magnitudes = np.linalg.norm(H, axis=2)

            yield B_magnitudes, H_magnitudes
        
    gen = generator(sources)
    results = []

    for _ in range(loop):
        results.append(next(gen))

    B_res, H_res = results[-1]  # Get the last yielded value (based on loop count)

    # print the results
    print("Here are the B magnitudes:\n\n" + str(B_res) + "\n\nand here are the H magnitudes:\n\n" + str(H_res) + "\n")

    # Create display object
    plot_field(sources)

"""loop_1 = magpy.current.Loop(
    current=1, 
    diameter=4,
    position=[0, 0, 5],
    )
loop_2 = magpy.current.Loop(
    current=1, 
    diameter=4, 
    position=[0, 0, -5],
    )
loop_3 = magpy.current.Loop(
    current=1, 
    diameter=4, 
    position=[0, 5, 0],
    orientation=R.from_euler("x", 90, degrees=True)
    )
loop_4 = magpy.current.Loop(
    current=1, 
    diameter=4, 
    position=[0, -5, 0],
    orientation=R.from_euler("x", 90, degrees=True)
    )

plot_magnitude(loop_1, loop_2, loop_3, loop_4, loop=1)"""