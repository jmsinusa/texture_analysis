'''
Created on 21 Dec 2015

@author: james
'''
import unittest
import second_attempt as sa
import numpy as np


class Test(unittest.TestCase):


    def test_linear_interp1(self):
        '''c = 0, m = 1'''
        m = 1
        c = 0
        x1 = 1
        x2 = 2
        y1 = x1 * m + c
        y2 = x2 * m + c
        testx = 3
        testy = testx * m + c
        calculatedy = sa.linear_interp(x1, y1, x2, y2, testx)
        self.assertEqual(testy, calculatedy)

    def test_linear_interp2(self):
        '''c = 0, m = 1'''
        m = 2
        c = 3
        x1 = 3
        x2 = 5
        y1 = x1 * m + c
        y2 = x2 * m + c
        testx = 2
        testy = testx * m + c
        calculatedy = sa.linear_interp(x1, y1, x2, y2, testx)
        self.assertEqual(testy, calculatedy)

    def test_linear_interp3(self):
        '''c = 0, m = 1'''
        m = -23
        c = -234.234
        x1 = -234.34
        x2 = -23.9
        y1 = x1 * m + c
        y2 = x2 * m + c
        testx = 17.3
        testy = testx * m + c
        calculatedy = sa.linear_interp(x1, y1, x2, y2, testx)
        self.assertAlmostEqual(testy, calculatedy, places = 3)
    
    def test_create_grid(self):
        '''Create the grid'''
        start = np.array((1, 1))
        stop = np.array((5, 6))
        ncells = np.array((4, 5))
        desiredx = np.array((1, 2, 3, 4, 5))
        desiredy = np.array((1, 2, 3, 4, 5, 6))
        (outx, outy) = sa.create_grid(start, stop, ncells)
        for ii in range(len(desiredx)):
            self.assertEqual(desiredx[ii], outx[ii])
        for ii in range(len(desiredy)):
            self.assertEqual(desiredy[ii], outy[ii])

    def test_create_grid2(self):
        '''Create the grid'''
        start = np.array((1, -1))
        stop = np.array((5, -6))
        ncells = np.array((4, -5))
        desiredx = np.array((1, 2, 3, 4, 5))
        desiredy = np.array((-1, -2, -3, -4, -5, -6))
        (outx, outy) = sa.create_grid(start, stop, ncells)
        for ii in range(len(desiredx)):
            self.assertEqual(desiredx[ii], outx[ii])
        for ii in range(len(desiredy)):
            self.assertEqual(desiredy[ii], outy[ii])
    
    def test_create_grid_idx(self):
        '''Create an index for the grid'''
        xvals = np.array((1, 2, 3))
        yvals = np.array((2, 3, 4))
        ncells = np.array((2, 2))
        outidx = sa.create_grid_idx(xvals, yvals, ncells)
        desiredcell = (2, 2, 3, 3)
        desiredcell_no = 1
        for ii in range(len(desiredcell)):
            self.assertEqual(desiredcell[ii], outidx[desiredcell_no][ii])

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_calculate_linear_factors1']
    unittest.main()