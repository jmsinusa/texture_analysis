'''
Created on 20 Dec 2015

@author: james
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import exceptions
import matplotlib.patches as patches
import matplotlib.colors as colors
import matplotlib.cm as colormap
from matplotlib.path import Path

def read_jpg(filename = '/Users/james/workspace/texture_analysis/DT.jpg'):
    '''Read jpg file into numpy array'''
    img = mpimg.imread(filename)
    img = np.mean(img, axis = 2)
    #plt.imshow(img, cmap = 'hot')
    #plt.show()
    return img

def linear_interp(x1, y1, x2, y2, testx):
    '''for two known points on a line, find the y for an unknown x.
    There's probably a numpy function that does this!'''
    grad = (y2 - y1) / (x2 - x1)
    const = y1 - grad * x1
    return testx * grad + const

def create_grid(start, stop, ncells):
    '''Create a grid
    start: numpy two-element array of start (x, y)
    stop: numpy two-element array of stop (x, y)
    ncells: numpy two-element array of ncells (x, y)'''
    xvals = np.linspace(start[0], stop[0], abs(ncells[0]) + 1)
    yvals = np.linspace(start[1], stop[1], abs(ncells[1]) + 1)
    return xvals, yvals

def create_grid_idx(xvals, yvals, ncells):
    '''Create an index using a dictionary
    ind{cell_no} = (xstart, ystart, xend, yend)'''
    ind = {}
    cell_no = 0
    for yy in range(int(abs(ncells[1]))):
        for xx in range(int(abs(ncells[0]))):
            ind[cell_no] = (xvals[xx], yvals[yy], xvals[xx + 1], yvals[yy + 1])
            cell_no += 1
    return ind

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = colormap.ScalarMappable(norm=color_norm, cmap='hsv') 
    return scalar_map

class texture(object):
    '''
    Create grids and conduct texture analysis
    '''

    def __init__(self, data):
        '''
        Constructor
        '''
        self.raw_img = data
        self.corners = []
    
    def select_grid_corners(self):
        '''Select the SW and NE corners of a box
        return [[x1,y1,xdata1,ydata1], [x2,y2,xdata2,ydata2]]
        x,y: positions on graph
        xdata,ydata: postitions in data matrix'''
        contin = True
        while contin == True:
            if len(self.corners) > 2:
                self.corners = []
                print 'Corners reset'
            fig = plt.figure() 
            ax = fig.add_subplot(111)
            ax.imshow(self.raw_img, cmap = 'hot')
            cid = fig.canvas.mpl_connect('button_press_event', self._onclick)
            plt.show()  
            done = self._2corners_test()
            if done:
                contin = False
            fig.canvas.mpl_disconnect(cid)
        print self.corners
        
    def groundtruth_select_corners(self, gt_option = 1):
        '''Load the corners used for ground truthing'''
        if gt_option == 1:
            self.corners = [[291, 259, 289.27083333333337, 514.89583333333337], 
                            [402, 357, 619.95833333333348, 222.9375]]
        if gt_option == 2:
            self.corners = [[316, 313, 363.75, 354.02083333333337],
                            [319, 319, 372.6875, 336.14583333333337]]
        if gt_option == 3:
            #starting on left eye
            self.corners = [[315, 312, 360.77083333333337, 357.0], 
                            [375, 424, 539.52083333333326, 23.333333333333485]]
        else:
            raise exceptions.NotImplementedError('The selected gt option (%s) has not been implemented'% gt_option)
        print "  Ground Truth mode.\nGT %s corner positions loaded:"% gt_option
        for cc in self.corners:
            print "    Corner display (%i, %i), data (%6.2f, %6.2f)" % (cc[0], cc[1], cc[2], cc[3])
    
    def create_grid(self, xpixels = 8, ypixels = 8):
        '''Create a grid, using self.corners, of size xpixels, ypixels.
        Report number of cells. Expand in x and y direction to ensure integer numer of cells.'''
        print "Creating grid of cells, each (%i, %i) pixels"% (xpixels, ypixels)
        self.cellsize = np.array((xpixels, ypixels))
        
        #Calculate the size of the region selected in both data (pixels) and plot coordinates
        initial_pixels = [None, None]
        initial_pixels[0] = np.array((self.corners[0][2], self.corners[0][3]))
        initial_pixels[1] = np.array((self.corners[1][2], self.corners[1][3]))
        
        initial_plotpos = [None, None]
        initial_plotpos[0] = np.array((self.corners[0][0], self.corners[0][1]))
        initial_plotpos[1] = np.array((self.corners[1][0], self.corners[1][1]))
        
        #save these for futue conversions to/from plotted image
        self.initial_pixels = initial_pixels
        self.initial_plotpos = initial_plotpos

        #Now find the number of cells required, moving the second points if 
        #  necessary to make this an integer.
        #Round initial_pixels
        rounded_pixels = [None, None]
        rounded_pixels[0] = np.round(initial_pixels[0])
        rounded_pixels[1] = np.round(initial_pixels[1])
        
        rounded_pixel_size = rounded_pixels[1] - rounded_pixels[0]
        excess = np.remainder(rounded_pixel_size, self.cellsize)
        expandby = self.cellsize - excess
        print "  The selected area is being expanded by (%i, %i) pixels."% (expandby[0], expandby[1])
        rounded_pixels[1] = rounded_pixels[1] + expandby
        rounded_pixel_size = rounded_pixels[1] - rounded_pixels[0]
        
        ncells = rounded_pixel_size / self.cellsize
        self.ncells = ncells
        self.ncells_abs = int(np.prod(abs(ncells)))
        print "  Creating (%i, %i) grid of %i cells..."% (abs(ncells[0]), abs(ncells[1]), np.prod(abs(ncells)))

        #Create grid
        self.gridxvals, self.gridyvals = create_grid(rounded_pixels[0], rounded_pixels[1], ncells)
        self.grididx = create_grid_idx(self.gridxvals, self.gridyvals, ncells)
        
    def display_grid(self):
        '''Display the grid on the image'''
        #setup the known points for finding the cell bounding boxes
        xpixel1 = self.initial_pixels[0][0]
        xpixel2 = self.initial_pixels[1][0]
        xplot1 = self.initial_plotpos[0][0]
        xplot2 = self.initial_plotpos[1][0]
        
        ypixel1 = self.initial_pixels[0][1]
        ypixel2 = self.initial_pixels[1][1]
        yplot1 = self.initial_plotpos[0][1]
        yplot2 = self.initial_plotpos[1][1]
        
        #create a colourmap
        plt_alpha = 0.2
        colourmap = get_cmap(self.ncells_abs)
        cell_indexes = range(self.ncells_abs)
        colourindexes = np.random.choice(cell_indexes, size = len(cell_indexes), replace = False)
        colours = colourmap.to_rgba(colourindexes, alpha = plt_alpha)
     
     
        #TEST RECTANGLE
        
        #verts = [(xplot1, yplot1), (xplot1, yplot2), (xplot2, yplot2), (xplot2, yplot1), (xplot1, yplot1)]
        verts = [(xpixel1, ypixel1), (xpixel1, ypixel2), (xpixel2, ypixel2), (xpixel2, ypixel1), (xpixel1, ypixel1)]
        
        codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
        path = Path(verts, codes)
        
        fig = plt.figure() 
        ax = fig.add_subplot(111)
        ax.imshow(self.raw_img, cmap = 'hot')
        shp = patches.PathPatch(path, lw=0.5, facecolor = 'none')
        ax.add_patch(shp)
#         rect = patches.Rectangle((startx, starty), width, height, 
#                                  edgecolor = colours[0], linewidth = 1, facecolor = colours[0],
#                                  hatch = '/')
        plt.show()      
        #display image
        
        fig = plt.figure() 
        ax = fig.add_subplot(111)
        ax.imshow(self.raw_img, cmap = 'hot')
        for cell_no in range(self.ncells_abs):
            this_cell = self.grididx[cell_no]
            startx = this_cell[0]
            starty = this_cell[1]
            width = self.cellsize[0]
            height = self.cellsize[1]
            rect = patches.Rectangle((startx, starty), width, height, 
                                     edgecolor = 'black', linewidth = 1, facecolor = colours[cell_no])
            ax.add_patch(rect)
        plt.show()          
        
        xtest_pixel = self.initial_pixels[0][0]
        xfound_plot = linear_interp(xpixel1, xplot1, xpixel2, xplot2, xtest_pixel)


        

        
        #DEBUG - zero data within this area
        pass
        
        
    
    def _2corners_test(self):
        '''Check that two, and only two, corners have been selected'''
        try:
            assert len(self.corners) == 2
        except:
            print "You must select two corners. Try again."
            return False
        else:    
            return True        
        

    def _onclick(self, event):
        print 'Corner selected at (%i, %i)'% (round(event.xdata), round(event.ydata)) 
        self.corners.append([event.x, event.y, event.xdata, event.ydata])
     
if __name__ == '__main__':
    data = read_jpg()
    S = texture(data)
    S.select_grid_corners()
    #S.groundtruth_select_corners(gt_option = 3)
    S.create_grid()
    S.display_grid()