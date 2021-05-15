from HilbertCurve import hilbert_3D_curves
from lempel_ziv_77 import lz77
from multiprocessing import Pool
from statistics import mean
from pathlib import Path
import numpy as np

class InterlacedTime():

    def __init__(self, order, shuffles = 2):
        self.dim = 3
        self.order = order
        self.num_shuffles = shuffles
        self.hilbert_curves = hilbert_3D_curves(order)


    def isotropicQ(self, data):

        if type(data) is not np.ndarray:
            raise TypeError('Data must be a numpy array, got {}.'.format(type(data)))
        elif data.ndim != self.dim:
            raise ValueError('Dimensions of numpy-data-array must be %d, got %d' %(self.dim, data.ndim))
        elif data.shape[1:] != (1 << self.order, ) * (self.dim - 1):
            raise ValueError('...')
        elif data.shape[0] % data.shape[1] != 0:
            raise ValueError('...')

        self.data = data

        pool = Pool(len(self.hilbert_curves))
        qs = pool.map_async(self.q_order, self.hilbert_curves)
        pool.close()
        pool.join()
        
        return mean([q[0] for q in qs.get()]), mean([q[1] for q in qs.get()])


    def q_order(self, hilbert_curve):

        ## BAD CODE AHEAD!
        ## These next five lines has been hamared into the keyboard – it work but it ain't pretty!
        x0, x1, _ = self.data.shape
        hilbert_scan = np.empty((x0 // x1, x1**3), dtype=int)
        for i, cube in enumerate(np.split(self.data, x0 // x1, axis=0)):
            hilbert_scan[i] = np.fromiter((cube[point] for point in hilbert_curve), dtype=int)
        hilbert_scan = hilbert_scan.flatten()

        def list2string(mylist):
            return ''.join(map(str, mylist.tolist()))

        def cid(hscan):
            C = lz77(list2string(hscan))
            L = len(hscan)
            return (C*np.log2(C) + 2*C*np.log2(L/C)) / L

        def cid_shuffles(hscan):
            shuffles = np.empty(self.num_shuffles)
            rng = np.random.default_rng()
            for i in range(self.num_shuffles):
                rng.shuffle(hscan)
                shuffles[i] = cid(hscan)
            return np.mean(shuffles)

        return cid(hilbert_scan), 1 - cid(hilbert_scan) / cid_shuffles(hilbert_scan)


def import_data(path):
    data = np.load(str(path))
    L = data.shape[1]
    window = slice(L//4, 3*L//4)
    return data[:, window, window]
    
if __name__ == '__main__':
    ## TODO :: rewrite this to run i parallel for faster runtime.
    pathlist = Path('outputs/lattice').glob('*.npy')
    temps = np.load('outputs/output_.npy')[0]
    it = InterlacedTime(order=6, shuffles=4)
    cid = []
    Q = []
    for path in sorted(pathlist):
        data = import_data(path)
        a, b = it.isotropic_q_order(data)
        cid.append(a)
        Q.append(b)
    np.save('outputs/shuffle'+str(it.num_shuffles), [temps, cid, Q])