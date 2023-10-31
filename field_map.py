import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy
from scipy.spatial.transform import Rotation as R
from scipy.optimize import lsq_linear
from scipy.ndimage import gaussian_filter

import plot_field

class field_map_generator:
    def __init__(self, low=-2, high=2):
        self.map = np.random.uniform(low, high, (256, 256))

    @staticmethod
    def generator():
        while True:
            yield np.random.uniform(-2, 2, (256, 256))

    def __iter__(self):
        self.generator = self.generator()
    
    def __next__(self):
        return next(self.generator)
    
#generic field_map object

class field_map:
    
    def __init__(self, field_map=field_map_generator()): #field_map_generator object
        self.field_map = field_map.map
        self.generator = field_map_generator.generator()

    def field_map_generator(self, low=-2, high=2):
        self.field_map = next(self.generator)

    def corrected_field_map(self, *currents, sigma=30):
        assert all([current >= -2 and current <= 2 for current in currents]), "All currents must be between -2 and 2."

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
              "\n\n" +
              f"Standard Deviation of corrected field_map: {np.std(corrected_B)}" +
              "\n")

        plt.imshow(gaussfilt, cmap='gray', origin='lower')
        plot_field.plot_field(coll)
        plt.show()

#field_map object that implements lsq_linear to optimize currents

class field_map_lsq(field_map):

    def __init__(self, field_map=field_map_generator()):
        super().__init__(field_map)

    def more_corrected_field_map(self, sigma=30):
        Y, Z = np.mgrid[-5:5:256j, -5:5:256j]
        grid = np.stack([np.zeros((256, 256)), Y, Z], axis=2)

        coll = magpy.Collection(
            magpy.current.Loop(
                current=1, 
                diameter=5,
                position=[0, 0, 5],
            ),
            magpy.current.Loop(
                current=1, 
                diameter=5,
                position=[0, 0, -5],
            ),
            magpy.current.Loop(
                current=1, 
                diameter=5,
                position=[0, 5, 0],
                orientation=R.from_euler('x', 90, degrees=True),
            ),
            magpy.current.Loop(
                current=1, 
                diameter=5,
                position=[0, -5, 0],
                orientation=R.from_euler('x', 90, degrees=True),
            ),
        )

        B1 = coll[0].getB(grid)
        B2 = coll[1].getB(grid)
        B3 = coll[2].getB(grid)
        B4 = coll[3].getB(grid)

        A = np.stack([B1[:, :, 2], B2[:, :, 2], B3[:, :, 2], B4[:, :, 2]], axis=2).reshape(-1, 4)
        X = lsq_linear(A, self.field_map.reshape(-1), bounds=(-2, 2)).x
        
        print("\nHere are the currents that optimize the correction of the inputted field map:\n\n" + 
              str(X) +  
              "\n")
        coll = magpy.Collection(
            magpy.current.Loop(
                current=X[0], 
                diameter=5,
                position=[0, 0, 5],
            ),
            magpy.current.Loop(
                current=X[1], 
                diameter=5,
                position=[0, 0, -5],
            ),
            magpy.current.Loop(
                current=X[2], 
                diameter=5,
                position=[0, 5, 0],
                orientation=R.from_euler('x', 90, degrees=True),
            ),
            magpy.current.Loop(
                current=X[3], 
                diameter=5,
                position=[0, -5, 0],
                orientation=R.from_euler('x', 90, degrees=True),
            ),
        )

        B = coll.getB(grid)
        print("\nHere are the B magnitudes produced by the optimal currents:\n\n" + 
              str(B) +  
              "\n")
        
        corrected_B = self.field_map - B[:, :, 2]
        gaussfilt = gaussian_filter(corrected_B, sigma=sigma)
        print("\nAnd here is the corrected field_map using the loop collection:\n\n" + 
              str(corrected_B) +  
              "\n\n" +
              f"Standard Deviation of corrected field_map: {np.std(corrected_B)}" +
              "\n")
        
        plt.imshow(np.abs(gaussfilt), cmap='coolwarm', origin='lower')
        plt.show()

#test class

class test:
    @staticmethod
    def basic_test():
        fm = field_map()
        fm.corrected_field_map(1, 1, 1, 1, sigma=29)

    @staticmethod
    def advanced_test():
        fm = field_map()
        fm.field_map_generator()
        fm.corrected_field_map(1, 1, 1, 1)
    
    @staticmethod
    def multiple_regeneration_test():
        fm = field_map()
        for _ in range(5):
            fm.field_map_generator()
            fm.corrected_field_map(1, 1, 1, 1,)

    @staticmethod
    def lsq_basic_test():
        map = field_map_lsq()
        map.more_corrected_field_map()

    @staticmethod
    def lsq_advanced_test():
        map = field_map_lsq()
        map.field_map_generator()
        map.more_corrected_field_map()

    @staticmethod
    def lsq_multiple_regeneration_test():
        map = field_map_lsq()
        for _ in range(5):
            map.field_map_generator()
            map.more_corrected_field_map()

#Test
#test().basic_test()
#test().advanced_test()
#test().multiple_regeneration_test()

test().lsq_basic_test()
#test().lsq_advanced_test()
#test().lsq_multiple_regeneration_test()
