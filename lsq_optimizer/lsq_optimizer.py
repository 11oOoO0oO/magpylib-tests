import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy
from scipy.spatial.transform import Rotation as R
from scipy.optimize import lsq_linear
from scipy.ndimage import gaussian_filter

class lsq_optimizer:
    
    def __init__(self):
        self.map = None
        
    def setMap(self, filename):
        from pydicom import dcmread

        ds = dcmread(filename)
        arr = ds.pixel_array
        self.map = arr

        plt.imshow(arr, cmap="gray")
        plt.show()

    def generateMap(self, target_area=(256, 256)):
        self.map = np.random.uniform(-100, 100, target_area)
    
    def generateZeroesMap(self, target_area=(256, 256)):
        self.map = np.zeros(target_area)

    def generateOnesMap(self, target_area=(256, 256)):
        self.map = np.ones(target_area)

    def generateReverseMap(self, target_area_x=128):
        X, Y, Z = np.mgrid[-5:5:256j, -5:5:256j, -5:5:256j]
        grid = np.stack([X, Y, Z], axis=3)

        coll = magpy.Collection(
            magpy.current.Loop(
                current=1, 
                diameter=2.5,
                position=[0, 0, 5],
            ),
            magpy.current.Loop(
                current=1, 
                diameter=2.5,
                position=[0, 0, -5],
            ),
            magpy.current.Loop(
                current=1, 
                diameter=2.5,
                position=[0, 5, 0],
                orientation=R.from_euler('x', 90, degrees=True),
            ),
            magpy.current.Loop(
                current=1, 
                diameter=2.5,
                position=[0, -5, 0],
                orientation=R.from_euler('x', 90, degrees=True),
            ),
            magpy.current.Loop(
                current=1, 
                diameter=2.5,
                position=[5, 0, 0],
                orientation=R.from_euler('z', 90, degrees=True),
            ),
            magpy.current.Loop(
                current=1, 
                diameter=2.5,
                position=[-5, 0, 0],
                orientation=R.from_euler('z', 90, degrees=True),
            )
        )

        B1 = (coll[0].getB(grid) * (42.58) * (10 ** 3))[target_area_x, :, :, 2]
        B2 = (coll[1].getB(grid) * (42.58) * (10 ** 3))[target_area_x, :, :, 2]
        B3 = (coll[2].getB(grid) * (42.58) * (10 ** 3))[target_area_x, :, :, 2]
        B4 = (coll[3].getB(grid) * (42.58) * (10 ** 3))[target_area_x, :, :, 2]
        B5 = (coll[4].getB(grid) * (42.58) * (10 ** 3))[target_area_x, :, :, 2]
        B6 = (coll[5].getB(grid) * (42.58) * (10 ** 3))[target_area_x, :, :, 2]

        self.map = B1 + B2 + B3 + B4 + B5 + B6

    def optimizer(self, target_area_x=128):
        X, Y, Z = np.mgrid[-5:5:256j, -5:5:256j, -5:5:256j]
        grid = np.stack([X, Y, Z], axis=3)

        coll = magpy.Collection(
            magpy.current.Loop(
                current=1, 
                diameter=2.5,
                position=[0, 0, 5],
            ),
            magpy.current.Loop(
                current=1, 
                diameter=2.5,
                position=[0, 0, -5],
            ),
            magpy.current.Loop(
                current=1, 
                diameter=2.5,
                position=[0, 5, 0],
                orientation=R.from_euler('x', 90, degrees=True),
            ),
            magpy.current.Loop(
                current=1, 
                diameter=2.5,
                position=[0, -5, 0],
                orientation=R.from_euler('x', 90, degrees=True),
            ),
            magpy.current.Loop(
                current=1, 
                diameter=2.5,
                position=[5, 0, 0],
                orientation=R.from_euler('z', 90, degrees=True),
            ),
            magpy.current.Loop(
                current=1, 
                diameter=2.5,
                position=[-5, 0, 0],
                orientation=R.from_euler('z', 90, degrees=True),
            )
        )

        B1 = (coll[0].getB(grid) * (42.58) * (10 ** 3))[target_area_x, :, :, 2]
        B2 = (coll[1].getB(grid) * (42.58) * (10 ** 3))[target_area_x, :, :, 2]
        B3 = (coll[2].getB(grid) * (42.58) * (10 ** 3))[target_area_x, :, :, 2]
        B4 = (coll[3].getB(grid) * (42.58) * (10 ** 3))[target_area_x, :, :, 2]
        B5 = (coll[4].getB(grid) * (42.58) * (10 ** 3))[target_area_x, :, :, 2]
        B6 = (coll[5].getB(grid) * (42.58) * (10 ** 3))[target_area_x, :, :, 2]

        
        A = np.stack([B1, B2, B3, B4, B5, B6], axis=2).reshape(-1, 6)
        X = lsq_linear(A, self.map.reshape(-1)).x
        
        print("\nHere are the currents that optimize the correction of the inputted field map:\n\n" + 
              str(X) +  
              "\n")
        
        self.lsq_currents = X

# Test
        
opt = lsq_optimizer()
opt.generateReverseMap()
opt.optimizer()
