'''
Created on 21 Dec 2015

@author: james
'''
import unittest
import third_attempt as sa
import numpy as np


class Test(unittest.TestCase):

    
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

    def test_textures(self):
        '''Test blank white image'''
        desired = [255, 0, 0, 0, 1, 0]
        data = 255 * np.ones(shape = (1000, 1000))
        S = sa.texture(r'/Users/james/blog/20150918_republicanPCA/cropped_images')
        S.raw_img = data
        S.data_shape = S.raw_img.shape
        #S.select_grid_corners()
        S.groundtruth_select_corners(gt_option = 1)
        S.create_grid(xpixels = 60, ypixels = 60)
        #S.display_grid(showarea = False)
        #S.measure_texture(bins = 16)
        cell = S._getcell(data, 2)
        out = S._analyse_texture(cell, bins = 256)
        for ii in range(len(desired)):
            self.assertAlmostEqual(desired[ii], out[ii], places = 2)
    
    def test_centre_data(self):
        '''Centre 5 x 6 array'''
        data = np.array([[1, 2, 3, 4, 5, 6], [2, 4, 6, 8, 10, 12], [3, 6, 9, 12, 15, 18]])
        desired = np.array([[-1, -2, -3, -4, -5, -6], [0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6]])
        out = sa.calc_centre_data(data)
        for ii in range(out.shape[0]):
            for jj in range(out.shape[1]):
                self.assertEqual(out[ii, jj], desired[ii, jj])

    def test_norm_data(self):
        '''Centre 5 x 6 array'''
        data = np.array([[1, 2, 3, 4, 5, 6], [2, 4, 6, 8, 10, 12], [3, 6, 9, 12, 15, 18]])
        desired = np.array([[ 1.22474487,  1.22474487,  1.22474487,  1.22474487,  1.22474487,  1.22474487],
                            [ 2.44948974,  2.44948974,  2.44948974,  2.44948974,  2.44948974,  2.44948974],
                            [ 3.67423461,  3.67423461,  3.67423461,  3.67423461,  3.67423461,  3.67423461]])
        out = sa.calc_norm_data(data)
        for ii in range(out.shape[0]):
            for jj in range(out.shape[1]):
                self.assertAlmostEqual(out[ii, jj], desired[ii, jj])
    
    def test_quantise_data(self):
        '''Quantise'''
        data = np.array([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 1]])
        bins = 3
        desired = np.array([[1, 1, 3.5, 3.5, 6, 6], [1, 3.5, 3.5, 6, 6, 1]])
        out = sa.quantise_data(data, bins)
        for ii in range(out.shape[0]):
            for jj in range(out.shape[1]):
                self.assertEqual(out[ii, jj], desired[ii, jj])        
    
    def test_compare2bool(self):
        '''Compare two boolean matrixes'''
        ar = np.array([[True, False, True, False], [False, True, False, True]])
        gt = np.array([[False, False, True, False], [True, True, False, False]])
        d1 = 2
        d2 = 2
        d3 = 1
        d4 = 3
        (o1, o2, o3, o4) = sa.compare_2bool(ar, gt)
        self.assertEqual(d1, o1)
        self.assertEqual(d2, o2)
        self.assertEqual(d3, o3)
        self.assertEqual(d4, o4)
    


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_calculate_linear_factors1']
    unittest.main()