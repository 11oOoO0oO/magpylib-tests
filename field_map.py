import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter

import plot_field

class field_map_generator:
    def __init__(self, low=-2, high=2):
        self.map = np.random.uniform(low, high, (256, 256))

    @staticmethod
    def generator(low=-2, high=2):
        yield np.random.uniform(low, high, (256, 256))

    def __iter__(self):
        self.generator = self.generator()
    
    def __next__(self):
        return next(self.generator)

class field_map:
    
    def __init__(self, field_map=field_map_generator()):
        self.field_map = field_map.map
        self.generator = iter(self.field_map)

    def field_map_generator(self, low=-2, high=2):
        self.field_map = next(self.generator)

    def corrected_field_map(self, *currents, sigma=30):
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
        gaussfilt = gaussian_filter(corrected_B, sigma=sigma)
        print("\nAnd here is the corrected field_map using the loop collection:\n\n" + 
              str(corrected_B) +  
              "\n")

        plt.imshow(gaussfilt, cmap='gray')
        plot_field.plot_field(coll)
        plt.show()

class test:
    @staticmethod
    def basic_test():
        fm = field_map()
        fm.corrected_field_map(1, 1, 1, 1)

    @staticmethod
    def advanced_test():
        fm = field_map()
        fm.field_map_generator()
        fm.corrected_field_map(1, 1, 1, 1)
    
    @staticmethod
    def multiple_regeneration_test():
        fm = field_map()
        for i in range(10):
            fm.field_map_generator()
            fm.corrected_field_map(1, 1, 1, 1)

#Test
test().basic_test()
test().advanced_test()
test().multiple_regeneration_test()

        
    
