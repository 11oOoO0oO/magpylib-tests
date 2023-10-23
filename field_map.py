import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter

class field_map:
    
    def __init__(self, field_map=np.random.uniform(low=-2, high=2, size=(256, 256))):
        self.field_map = field_map 

    def field_map_generator(self, low=-2, high=2):
        self.field_map = np.random.uniform(low, high, (256, 256))

    def corrected_field_map(self, *currents):
        assert all([current > -2 and current < 2 for current in currents]), "All currents must be between -2 and 2."

        Y, Z = np.mgrid[-5:5:256j, -5:5:256j]
        grid = np.stack([np.zeros((256, 256)), Y, Z], axis=2)

        coll = magpy.Collection(
            magpy.current.Loop(
                current=currents[0], 
                diameter=5,
                position=[0, 0, 5],
            ),
            magpy.current.Loop(
                current=currents[1], 
                diameter=5,
                position=[0, 0, -5],
            ),
            magpy.current.Loop(
                current=currents[2], 
                diameter=5,
                position=[0, 5, 0],
                orientation=R.from_euler('x', 90, degrees=True),
            ),
            magpy.current.Loop(
                current=currents[3], 
                diameter=5,
                position=[0, -5, 0],
                orientation=R.from_euler('x', 90, degrees=True),
            ),
        )

        B = coll.getB(grid)
        print("\nHere are the B magnitudes for the loop collection:\n\n" + 
              str(B) +  
              "\n")

        corrected_B = self.field_map - B[:, :, 2]
        gaussfilt = gaussian_filter(corrected_B, sigma=30)
        print("\nAnd here is the corrected field_map using the loop collection:\n\n" + 
              str(corrected_B) +  
              "\n")

        plt.imshow(gaussfilt, cmap='gray')
        plt.show()

#Test
"""fm = field_map()
fm.field_map_generator()
fm.corrected_field_map(1, 1, 1, 1)"""

        
    
