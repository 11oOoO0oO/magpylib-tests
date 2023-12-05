import biot_savart as bs
from lsq_optimizer import lsq_optimizer

class simple_lsq_optimizer (lsq_optimizer):
    
    def __init__(self):
        super().__init__()

    def testOptimizeMap(self):

        self.generateCoil("base.txt", 1, 32, 0, 32, 0, 'x', 0)
        bs.write_target_volume("base.txt", "base", (64, 64, 64), (-32, -32, -32), 1, 1)
        fields, position = bs.read_target_volume("base")
        self.map = fields[:, :, :, 2] / 10000 * 42.58 * 100000
        bs.plot_fields(fields, position, which_plane="x", level=0, num_contours=50)

        self.coils = list()

        self.generateCoil("simple_text.txt", 1, 32, 0, 32, 0, 'x', 0)
        self.optimizeMap((64, 64, 64))

        bs.write_target_volume("simple_text.txt", "simple_test", (64, 64, 64), (-32, -32, -32), 1, 1)
        temp, _ = bs.read_target_volume("simple_test")
        bs.plot_fields(fields - temp, position, which_plane="x", level=0, num_contours=50)
        
#Test

test = simple_lsq_optimizer()
test.testOptimizeMap()

