'''
Created on 20 Dec 2015

@author: james
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import exceptions, warnings
import matplotlib.patches as patches
import matplotlib.colors as colors
import matplotlib.cm as colormap
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import operator
import os

def read_jpg(filename = r'/Users/james/blog/20150918_republicanPCA/cropped_images/Donald_Trump.jpg'):
    '''Read jpg file into numpy array'''
    img = mpimg.imread(filename)
    img = np.mean(img, axis = 2)
    #plt.imshow(img, cmap = 'hot')
    #plt.show()
    return img

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

def calc_centre_data(data):
    '''Centre the n x m matrix data,
    centre by subtracting the mean of each dimension'''
    means = np.mean(data, 0)
    data = data - means
    return data

def calc_norm_data(data):
    '''normalise each dimension so that its sd is one.'''
    sds = np.std(data, 0)
    data = np.divide(data, sds)
    return data

def order_means(data, cluster_nos):
    '''Order the cluster numbers in increasing values of the mean of the first dimension of data'''
    cluster_means = {}
    for ii in np.unique(cluster_nos):
        pos = cluster_nos == ii
        cluster_means[ii] = np.mean(data[pos, 0])
    sorted_cluster_means = sorted(cluster_means.items(), key = operator.itemgetter(1))
    sorted_cluster_nos = []
    for item in sorted_cluster_means:
        sorted_cluster_nos.append(item[0])
    return sorted_cluster_nos

def quantise_data(data, bins):
    '''Quantise numpy array data into bins number of bins'''
    data = data.astype(float)
    original_min = np.amin(data)
    original_max = np.amax(data)
    #remove offset
    data = data - np.amin(data)
    #scale to one
    data = data / np.amax(data)
    #multiply by number of bins - 1
    data = data * (bins - 1)
    #round to integer
    data = np.around(data, decimals = 0)
    #scale to one
    data = data / np.amax(data)
    #scale to original size
    data = data * (original_max - original_min) + original_min
    return data

class texture(object):
    '''
    Create grids and conduct texture analysis
    '''

    def __init__(self, images_dir, gt_dir = r'/Users/james/blog/20150918_republicanPCA/gt_dir'):
        '''
        Constructor
        '''
        #Get all images in directory images_dir
        #Check it's a directory first!
        if not os.path.isdir(images_dir):
            raise exceptions.IOError('Input %s must be a directory' %images_dir)
        if not os.path.isdir(gt_dir):
            raise exceptions.IOError('Input %s must be a directory' %gt_dir)
        files = os.listdir(images_dir)
        files.remove('.DS_Store')
        self.images_dir = images_dir
        self.images_files = files
        self.gt_dir = gt_dir
        #Read in first image
        self.raw_img = self.load_one_image()
        self.data_shape = self.raw_img.shape
        self.corners = []
        
    
    def load_one_image(self, imageno = 0, filename = None):
        '''Load one image to use for grid building.
        If filename, load images_dir/filename
        else, load image number imageno from images_dir'''
        if not filename:
            filename = self.images_files[imageno]
        filepath = self.images_dir + os.sep + filename
        image_data = read_jpg(filepath)
        return image_data
           
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
            plt.title('Select two points')
            cid = fig.canvas.mpl_connect('button_press_event', self._onclick)
            plt.show()  
            done = self._2corners_test()
            if done:
                contin = False
            fig.canvas.mpl_disconnect(cid)
        
    def groundtruth_select_corners(self, gt_option = 1):
        '''Load the corners used for ground truthing'''
        if gt_option == 1:
            self.corners = [[65.833333333333371, 958.79166666666674], [542.5, 374.875]]
        else:
            raise exceptions.NotImplementedError('The selected gt option (%s) has not been implemented'% gt_option)
        print "  Ground Truth mode.\nGT %s corner positions loaded:"% gt_option
        for cc in self.corners:
            print "    Corner (%6.2f, %6.2f)" % (cc[0], cc[1])
        self.gt_option = gt_option
    
    def create_grid(self, xpixels = 8, ypixels = 8):
        '''Create a grid, using self.corners, of size xpixels, ypixels.
        Report number of cells. Expand in x and y direction to ensure integer numer of cells.'''
        print "Creating grid of cells, each (%i, %i) pixels"% (xpixels, ypixels)
        self.cellsize = np.array((xpixels, ypixels))
        
        #Calculate the size of the region selected.
        initial_pixels = [None, None]
        initial_pixels[0] = np.array((self.corners[0][0], self.corners[0][1]))
        initial_pixels[1] = np.array((self.corners[1][0], self.corners[1][1]))
        
        
        #save these for future conversions to/from plotted image
        self.initial_pixels = initial_pixels

        #Now find the number of cells required, moving the second points if 
        #  necessary to make this an integer.
        #Round initial_pixels
        rounded_pixels = [None, None]
        rounded_pixels[0] = np.round(initial_pixels[0])
        rounded_pixels[1] = np.round(initial_pixels[1])
        
        rounded_pixel_size = rounded_pixels[1] - rounded_pixels[0]
        excess = np.remainder(abs(rounded_pixel_size), self.cellsize)
        expandby = self.cellsize - excess
        print "  The selected area is being expanded by (%i, %i) pixels."% (expandby[0], expandby[1])
        shapedir = rounded_pixel_size / abs(rounded_pixel_size)
        rounded_pixels[1] - rounded_pixels[0]
        rounded_pixels[1] = rounded_pixels[1] + shapedir * expandby
        rounded_pixels[1] - rounded_pixels[0]
        
        rounded_pixel_size = rounded_pixels[1] - rounded_pixels[0]
        
        ncells = rounded_pixel_size / self.cellsize
        self.ncells = ncells
        self.ncells_abs = int(np.prod(abs(ncells)))
        print "  Creating (%i, %i) grid of %i cells..."% (abs(ncells[0]), abs(ncells[1]), np.prod(abs(ncells)))

        #Create grid
        self.gridxvals, self.gridyvals = create_grid(rounded_pixels[0], rounded_pixels[1], ncells)
        self.grididx = create_grid_idx(self.gridxvals, self.gridyvals, ncells)
        
        #Create empty logical array for later groundtruthing
        self.gt_matrix = np.empty((len(self.images_files), self.ncells_abs), dtype = bool)
    
    def pixels_to_cellno(self, x, y):
        '''Convert a given x,y into a cell number'''
        #1. Check that x and y are within the bounds of the cell
        if x < min(self.gridxvals):
            raise exceptions.IOError('This point is outside of the grid')
        if x > max(self.gridxvals):
            raise exceptions.IOError('This point is outside of the grid')
        if y < min(self.gridyvals):
            raise exceptions.IOError('This point is outside of the grid')
        if y > max(self.gridyvals):
            raise exceptions.IOError('This point is outside of the grid')
        
        #2. Find the appropriate cell index
        xbig = min(self.gridxvals[self.gridxvals > x])
        xsmall = max(self.gridxvals[self.gridxvals < x])
        ybig = min(self.gridyvals[self.gridyvals > y])
        ysmall = max(self.gridyvals[self.gridyvals < y])
        
        if self.ncells[0] > 0:
            xpos = np.where(self.gridxvals == xsmall)[0]
        elif self.ncells[0] < 0:
            xpos = np.where(self.gridxvals == xbig)[0]
        if self.ncells[1] > 0:
            ypos = np.where(self.gridyvals == ysmall)[0]
        elif self.ncells[1] < 0:
            ypos = np.where(self.gridyvals == ybig)[0]
        cell_no = ypos * abs(self.ncells[0]) + xpos
        return int(cell_no)
    
    def display_grid(self, showarea = True):
        '''Display the grid on the image'''
        #setup the known points for finding the cell bounding boxes
        x1 = self.initial_pixels[0][0]
        x2 = self.initial_pixels[1][0]
        y1 = self.initial_pixels[0][1]
        y2 = self.initial_pixels[1][1]
        
        #create a colourmap
        plt_alpha = 0.4
        colourmap = get_cmap(self.ncells_abs)
        cell_indexes = range(self.ncells_abs)
        colourindexes = np.random.choice(cell_indexes, size = len(cell_indexes), replace = False)
        colours = colourmap.to_rgba(colourindexes, alpha = plt_alpha)
     
     
        #Show selected area
        if showarea:
            fig = plt.figure() 
            ax = fig.add_subplot(111)
            ax.imshow(self.raw_img, cmap = 'hot')
            width = self.initial_pixels[1][0] - self.initial_pixels[0][0]
            height = self.initial_pixels[1][1] - self.initial_pixels[0][1]
            rect = patches.Rectangle(self.initial_pixels[0], width, height, lw=1.5, 
                                     edgecolor = 'red', facecolor = 'none', ls = 'dashed')
            ax.add_patch(rect)
            plt.title('The area selected')
            plt.show()      
        
        #display grid squares
        
        fig = plt.figure() 
        ax = fig.add_subplot(111)
        ax.imshow(self.raw_img, cmap = 'hot')
        for cell_no in range(self.ncells_abs):
            this_cell = self.grididx[cell_no]
            x1 = this_cell[0]
            y1 = this_cell[1]
            x2 = this_cell[2]
            y2 = this_cell[3]
            startx = min(x1, x2)
            starty = min(y1, y2)
            width = self.cellsize[0]
            height = self.cellsize[1]
            rect = patches.Rectangle((startx, starty), width, height, 
                                     edgecolor = 'black', linewidth = 0.5, facecolor = colours[cell_no])
            ax.add_patch(rect)
        plt.title('%i cells, each %i x %i pixels'% (self.ncells_abs, self.cellsize[0], self.cellsize[1]))
        plt.show()

    def display_bitdepth_image(self, image_no, bins = 16):
        '''Display an image using only bins number of bins'''
        try:
            assert(bins > 1)
        except AssertionError:
            print 'The number of bins must be greater than one.'
        
        image = self.load_one_image(image_no)
        
        #Reduce depth
        image_to_display = quantise_data(image, bins)
        
        #display
        fig = plt.figure()        
        ax = fig.add_subplot(111)
        ax.imshow(image_to_display, cmap = 'gray')
        plt.title('Image %i number of bins %i'% (image_no, bins))
        plt.show() 
    
    def measure_all_textures(self, bins = 16):
        '''Loop through all images and measure textures'''
        n_images = len(self.images_files)
        all_texture = np.empty((n_images, self.ncells_abs, 6))
        ii = 0
        for file_ in self.images_files:
            filename = self.images_dir + os.sep + file_
            image_data = read_jpg(filename)
            #check dimensions match first image
            assert image_data.shape == self.data_shape
            texture = self.measure_texture(image_data, bins = bins)
            all_texture[ii, :, :] = texture
            ii += 1
        #print all_texture[0,] #first image, all cells
        #print all_texture[:,0,] #all images, first cell 
        #print all_texture[:,0,0]# all images, mean of first cell
        self.all_textures = all_texture
    
    def measure_texture(self, image_data, bins = 16):
        '''Measure the texture in each of the cells of image image_data.
        Save self.texture_array'''
        texture_array = np.empty((self.ncells_abs, 6))
        for cell_no in range(self.ncells_abs):
            data = self._getcell(image_data, cell_no)
            texture_array[cell_no,] = self._analyse_texture(data, bins = bins) 
        return texture_array
    
    def _select_image_or_cell_data(self, image_no = None, cell_no = None, 
                                   norm_data = False, centre_data = False, pca_data = False):
        '''Select six-dimensional data from either image_no or cell_no
        centre_data: subtract mean from each dimension
        norm_data: divide each dimension by its standard deviation
        retrun data: numpy matrix ncells x 6'''
        if image_no == None and cell_no == None:
            raise exceptions.IOError('You must specify either an image number of a cell number')
        if image_no and cell_no:
            raise exceptions.IOError('You must specify only one of an image number of a cell number')
        if image_no != None:
            data = self.all_textures[image_no, :, :]
        if cell_no != None:
            data = self.all_textures[:, cell_no, :]
        if centre_data:
            data = calc_centre_data(data)
        if norm_data:
            data = calc_norm_data(data)
        if pca_data:
            pca = PCA()
            data = pca.fit_transform(data)
        return data        
    
    def cluster_cells(self, image_no = None, cell_no = None, method = None, 
                      norm_data = False, centre_data = False, pca_data = False, 
                      kmeans_k = 2, dbscan_eps = 2, dbscan_ms = 5):
        '''Select all cells in either image_no or cell_no and cluster using method
        self.raw_clusters - the cluster the cell is assigned to be the clusering algorithm
        self.class_results - TRUE if an anomoly, FLASE if not'''
        #1. Select the relevant data
        data = self._select_image_or_cell_data(image_no = image_no, cell_no = cell_no, 
                                               norm_data = norm_data, centre_data = centre_data, pca_data = pca_data)
        #2. cluster
        if method == 'rand':
            #Randomly assign points to clusters 0 to 5.
            #Anomalies are clusters 3, 4, 5
            cluster_results = np.random.randint(0, high = 6, size = data.shape[0])
            bin_results = cluster_results >= 3
        if method == 'manual':
            #Manually select points that are anomalies. Anomalies are in cluster 1
            exceptions.NotImplementedError('Not implemented on mac due to mac specific error.')
        if method == 'kmeans':
            #Run k-means with k = kmeans_k and hope for the best
            #The cluster with the highest mean is taken as anomalous 
            kmeans = KMeans(n_clusters = kmeans_k, n_init = 30)
            cluster_results = kmeans.fit_predict(data)
            cluster_order = order_means(data, cluster_results)
            bin_results = cluster_results == cluster_order[-1]
        if method == 'dbscan':
            #Run dbscan with eps = dbscan_eps and min_samples = dbscan_ms
            #anomalies are those in cluster zero (ie no cluster)
            dbscan = DBSCAN(eps = dbscan_eps, min_samples = dbscan_ms)
            cluster_results = dbscan.fit_predict(data)
            cluster_results += 1 #so that the plotting colours work
            bin_results = cluster_results == 0
        else:
            raise exceptions.AttributeError('method %s not known.'% method)
        self.raw_clusters = cluster_results
        self.class_result = bin_results
        return bin_results

    def cluster_all_cells(self, method = None, 
                      norm_data = False, centre_data = False, pca_data = False, 
                      kmeans_k = 2, dbscan_eps = 2, dbscan_ms = 5):
        '''Cluster all cells by calling cluster_cells over each of the cells
        Save results into self.all_cell_class as matrix [n x m]
        self.all_cell_class[n, m] gives whether image n, cell m is an anomaly'''
        no_images = len(self.images_files)
        master_results = np.empty((len(self.images_files), self.ncells_abs), dtype = bool)
        #master_results[n, m] = anomaly on image n, cell m?
        for cell_no in range(self.ncells_abs):
            bin_results = self.cluster_cells(method = method, cell_no = cell_no, 
                                             norm_data = norm_data, centre_data = centre_data, 
                                             pca_data = pca_data, kmeans_k = kmeans_k, 
                                             dbscan_eps = dbscan_eps, dbscan_ms = dbscan_ms)
            master_results[:, cell_no] = bin_results
        self.all_cell_class = master_results     
    
    def plot_clusters_textures(self, image_no = None, cell_no = None,
                                norm_data = False, centre_data = False, pca_data = False):
        '''Plot the texture results for either an image or a cell,
        coloured by the cluster numbers from self.raw_clusters'''
        #1. Select the relevant data
        data = self._select_image_or_cell_data(image_no = image_no, cell_no = cell_no, 
                                               norm_data = norm_data, centre_data = centre_data, pca_data = pca_data)
        cluster_indx = np.unique(self.raw_clusters)
        print "DEBUG"
        print cluster_indx
        #2. Set colour palate
        plt_alpha = 1.0
        colourmap = get_cmap(len(cluster_indx) + 1)
        #Randomise colours - wise?
        #colourindexes = np.random.choice(cluster_indx, size = len(cell_indexes), replace = False)
        colourindexes = cluster_indx
        colours = colourmap.to_rgba(colourindexes, alpha = plt_alpha)
        print colours
        #2. Plot each cluster
        # Do some plotting
        
        if pca_data:
            f, axarr = plt.subplots(2, 2)
            #Band 1 vs band 2
            for cluster_no in cluster_indx:
                data_indx = self.raw_clusters == cluster_no
                band1 = data[data_indx, 0]
                band2 = data[data_indx, 1]
                axarr[0, 0].scatter(band1, band2, color = colours[cluster_no])
            axarr[0, 0].set_title('PCA 1 v PCA 2')
            
            #Band 2 vs Band 3
            for cluster_no in cluster_indx:
                data_indx = self.raw_clusters == cluster_no
                band2 = data[data_indx, 1]
                band3 = data[data_indx, 2]
                axarr[0, 1].scatter(band2, band3, color = colours[cluster_no])
            axarr[0, 1].set_title('PCA 2 v PCA 3')            
            
            #Band 2 vs Band 4
            for cluster_no in cluster_indx:
                data_indx = self.raw_clusters == cluster_no
                band2 = data[data_indx, 1]
                band4 = data[data_indx, 3]
                axarr[1, 1].scatter(band2, band4, color = colours[cluster_no])
            axarr[1, 1].set_title('PCA 2 v PCA 4')
            
            #Band 1 vs Band 3
            for cluster_no in cluster_indx:
                data_indx = self.raw_clusters == cluster_no
                band1 = data[data_indx, 0]
                band3 = data[data_indx, 2]
                axarr[1, 0].scatter(band1, band3, color = colours[cluster_no])
            axarr[1, 0].set_title('PCA 1 v PCA 3')
            
        else:
            #Non-PCA version
            f, axarr = plt.subplots(2, 2)
            #Mean vs SD
            for cluster_no in cluster_indx:
                data_indx = self.raw_clusters == cluster_no
                mean = data[data_indx, 0]
                sd = data[data_indx, 1]
                axarr[0, 0].scatter(mean, sd, color = colours[cluster_no])
            axarr[0, 0].set_title('mean v sd')
            
            #Smoothness vs third moment
            for cluster_no in cluster_indx:
                data_indx = self.raw_clusters == cluster_no
                smoothness = data[data_indx, 2]
                third_moment = data[data_indx, 3]
                axarr[0, 1].scatter(smoothness, third_moment, color = colours[cluster_no])
            axarr[0, 1].set_title('smoothness v third_moment')            
            
            #Smoothness vs third moment
            for cluster_no in cluster_indx:
                data_indx = self.raw_clusters == cluster_no
                uniformity = data[data_indx, 4]
                entropy = data[data_indx, 5]
                axarr[1, 1].scatter(uniformity, entropy, color = colours[cluster_no])
            axarr[1, 1].set_title('uniformity v entropy')
            
            #Mean vs entropy
            for cluster_no in cluster_indx:
                data_indx = self.raw_clusters == cluster_no
                mean = data[data_indx, 0]
                entropy = data[data_indx, 5]
                axarr[1, 0].scatter(mean, entropy, color = colours[cluster_no])
            axarr[1, 0].set_title('mean v entropy')

        plt.show()

    def plot_class_textures(self, image_no = None, cell_no = None, 
                            norm_data = False, centre_data = False, pca_data = False):
        '''Plot the texture results for either an image or a cell,
        coloured by the class from self.class_result'''
        #1. Select the relevant data
        data = self._select_image_or_cell_data(image_no = image_no, cell_no = cell_no, 
                                               norm_data = norm_data, centre_data = centre_data, pca_data = pca_data)
        #2. Plot each cluster
        # Do some plotting
        if pca_data:
            f, axarr = plt.subplots(2, 2)

            #Band 1 vs Band 2
            data_indx = self.class_result == True
            band1 = data[data_indx, 0]
            band2 = data[data_indx, 1]
            axarr[0, 0].scatter(band1, band2, color = 'red')
            data_indx = self.class_result == False
            band1 = data[data_indx, 0]
            band2 = data[data_indx, 1]
            axarr[0, 0].scatter(band1, band2, color = 'black')
            axarr[0, 0].set_title('PCA 1 v PCA 2')
            
            #Band 2 vs Band 3
            data_indx = self.class_result == True
            band2 = data[data_indx, 1]
            band3 = data[data_indx, 2]
            axarr[0, 1].scatter(band2, band3, color = 'red')
            data_indx = self.class_result == False
            band2 = data[data_indx, 1]
            band3 = data[data_indx, 2]
            axarr[0, 1].scatter(band2, band3, color = 'black')
            axarr[0, 1].set_title('PCA 2 v PCA 3')
                
            #Band 2 vs Band 4
            data_indx = self.class_result == True
            band2 = data[data_indx, 1]
            band4 = data[data_indx, 3]
            axarr[1, 1].scatter(band2, band4, color = 'red')
            data_indx = self.class_result == False
            band2 = data[data_indx, 1]
            band4 = data[data_indx, 3]
            axarr[1, 1].scatter(band2, band4, color = 'black')
            axarr[1, 1].set_title('PCA 2 v PCA 4')
            
            #Band 1 vs Band 4
            data_indx = self.class_result == True
            band1 = data[data_indx, 0]
            band3 = data[data_indx, 2]
            axarr[1, 0].scatter(band1, band3, color = 'red')
            data_indx = self.class_result == False
            band1 = data[data_indx, 0]
            band3 = data[data_indx, 2]
            axarr[1, 0].scatter(band1, band3, color = 'black')    
            axarr[1, 0].set_title('PCA 1 v PCA 3')
            
        else:
            #Not pca
            f, axarr = plt.subplots(2, 2)
            
            #Mean vs SD
            data_indx = self.class_result == True
            mean = data[data_indx, 0]
            sd = data[data_indx, 1]
            axarr[0, 0].scatter(mean, sd, color = 'red')
            data_indx = self.class_result == False
            mean = data[data_indx, 0]
            sd = data[data_indx, 1]
            axarr[0, 0].scatter(mean, sd, color = 'black')
            axarr[0, 0].set_title('mean v sd')
            
            #Smoothness vs third moment
            data_indx = self.class_result == True
            smoothness = data[data_indx, 2]
            third_moment = data[data_indx, 3]
            axarr[0, 1].scatter(smoothness, third_moment, color = 'red')
            data_indx = self.class_result == False
            smoothness = data[data_indx, 2]
            third_moment = data[data_indx, 3]
            axarr[0, 1].scatter(smoothness, third_moment, color = 'black')
            axarr[0, 1].set_title('smoothness v third_moment')       
                
            #Uniformity vs entropy
            data_indx = self.class_result == True
            uniformity = data[data_indx, 4]
            entropy = data[data_indx, 5]
            axarr[1, 1].scatter(uniformity, entropy, color = 'red')
            data_indx = self.class_result == False
            uniformity = data[data_indx, 4]
            entropy = data[data_indx, 5]
            axarr[1, 1].scatter(uniformity, entropy, color = 'black')
            axarr[1, 1].set_title('uniformity v entropy')
    
            #Mean vs entropy
            data_indx = self.class_result == True
            mean = data[data_indx, 0]
            entropy = data[data_indx, 5]
            axarr[1, 0].scatter(mean, entropy, color = 'red')
            data_indx = self.class_result == False
            mean = data[data_indx, 0]
            entropy = data[data_indx, 5]
            axarr[1, 0].scatter(mean, entropy, color = 'black')
            axarr[1, 0].set_title('mean v entropy')

        plt.show()
        
    def plot_raw_textures(self, image_no = None, cell_no = None, 
                          norm_data = False, centre_data = False, pca_data = False):
        '''Plot the texture results for either an image or a cell'''
        #1. Select the relevant data
        data = self._select_image_or_cell_data(image_no = image_no, cell_no = cell_no, 
                                               norm_data = norm_data, centre_data = centre_data, pca_data = pca_data)
        mean = data[:, 0]
        sd = data[:, 1]
        smoothness = data[:, 2]
        third_moment = data[:, 3]
        uniformity = data[:, 4]
        entropy = data[:, 5]

        # Do some plotting
        if pca_data:
            #Plot data - note that mean etc are now meaningless terms
            f, axarr = plt.subplots(2, 2)
            axarr[0, 0].scatter(mean, sd)
            axarr[0, 0].set_title('PCA 1 v PCA 2')
                
            axarr[0, 1].scatter(sd, smoothness)
            axarr[0, 1].set_title('PCA 2 v PCA 3')
            
            axarr[1, 1].scatter(sd, third_moment)
            axarr[1, 1].set_title('PCA 2 v PCA 4')
                
            axarr[1, 0].scatter(mean, smoothness)
            axarr[1, 0].set_title('PCA 1 v PCA 3')
        
        else:
            f, axarr = plt.subplots(2, 2)
            axarr[0, 0].scatter(mean, sd)
            axarr[0, 0].set_title('mean v sd')
                
            axarr[0, 1].scatter(smoothness, third_moment)
            axarr[0, 1].set_title('smoothness v third_moment')        
            
            axarr[1, 1].scatter(uniformity, entropy)
            axarr[1, 1].set_title('uniformity v entropy')
                
            axarr[1, 0].scatter(mean, entropy)
            axarr[1, 0].set_title('mean v entropy')
        plt.show()

    def show_anomalies(self, image_no = 0):
        '''Show anomalies plotted on top of image number image_no'''
        '''Display the grid on the image'''
        #setup the known points for finding the cell bounding boxes
        x1 = self.initial_pixels[0][0]
        x2 = self.initial_pixels[1][0]
        y1 = self.initial_pixels[0][1]
        y2 = self.initial_pixels[1][1]
        
        image_to_display = self.load_one_image(image_no)
        
        #display grid squares
   
        fig = plt.figure() 
        ax = fig.add_subplot(111)
        ax.imshow(image_to_display, cmap = 'gray')
        anom_count = 0
        for cell_no in range(self.ncells_abs):
            if self.all_cell_class[image_no, cell_no]:
                anom_count += 1
                this_cell = self.grididx[cell_no]
                x1 = this_cell[0]
                y1 = this_cell[1]
                x2 = this_cell[2]
                y2 = this_cell[3]
                startx = min(x1, x2)
                starty = min(y1, y2)
                width = self.cellsize[0]
                height = self.cellsize[1]
                rect = patches.Rectangle((startx, starty), width, height, 
                                         edgecolor = 'red', linewidth = 1.0, fill = False, facecolor = None, alpha = 0.6)
                ax.add_patch(rect)
                rect = patches.Rectangle((startx, starty), width, height, 
                                         edgecolor = None, linewidth = 0, fill = True, facecolor = 'red', alpha = 0.15)
                ax.add_patch(rect)
        plt.title('%i cells in this image are anomalous'% (anom_count))
        plt.show()  
    
    def show_image_cell(self, image_no, cell_no):
        '''Show image number image_no with cell number cell_no highlighted'''
        #setup the known points for finding the cell bounding boxes
        
        image_to_display = self.load_one_image(image_no)
        
        #display grid squares
        
        fig = plt.figure()        
        ax = fig.add_subplot(111)
        ax.imshow(image_to_display, cmap = 'gray')
        this_cell = self.grididx[cell_no]
        x1 = this_cell[0]
        y1 = this_cell[1]
        x2 = this_cell[2]
        y2 = this_cell[3]
        startx = min(x1, x2)
        starty = min(y1, y2)
        width = self.cellsize[0]
        height = self.cellsize[1]
        rect = patches.Rectangle((startx, starty), width, height, 
                                 edgecolor = 'blue', linewidth = 1.0, fill = False, facecolor = None, alpha = 0.6)
        ax.add_patch(rect)
        rect = patches.Rectangle((startx, starty), width, height, 
                                 edgecolor = None, linewidth = 0, fill = True, facecolor = 'blue', alpha = 0.15)
        ax.add_patch(rect)
        plt.title('Image %i cell %i'% (image_no, cell_no))
        plt.show()
    
    def record_all_groundtruth(self):
        '''Record ground truth (True / False) for each image'''
        #record ground truth in self.gt_matrix[image_no, cell_no] = True if anomaly
        for image_no in range(len(self.images_files)):
            self.gt_image(image_no)

    def gt_image(self, image_no):
        '''Display the grid on the image, allow points to be selected'''
        #record ground truth in self.gt_matrix[image_no, cell_no] = True if anomaly
        self.gt_cells = [] #empty list for storing ground truth points
        #display grid squares
        image_to_display = self.load_one_image(image_no)
        
        fig = plt.figure() 
        ax = fig.add_subplot(111)
        ax.imshow(image_to_display, cmap = 'gray')
        for cell_no in range(self.ncells_abs):

            this_cell = self.grididx[cell_no]
            x1 = this_cell[0]
            y1 = this_cell[1]
            x2 = this_cell[2]
            y2 = this_cell[3]
            startx = min(x1, x2)
            starty = min(y1, y2)
            width = self.cellsize[0]
            height = self.cellsize[1]
            rect = patches.Rectangle((startx, starty), width, height, 
                                     edgecolor = 'red', linewidth = 1.0, fill = False, facecolor = None, alpha = 0.6)
            ax.add_patch(rect)
            
            rect = patches.Rectangle((startx, starty), width, height, 
                                     edgecolor = None, linewidth = 0, fill = True, facecolor = 'red', alpha = 0.15)
            ax.add_patch(rect)

        plt.title('Select cells that contain activity')
        cid = fig.canvas.mpl_connect('button_press_event', self._gt_onclick)
        plt.show()  
        fig.canvas.mpl_disconnect(cid)
        print 'GT setup: Activity recorded for image %i in %i cells: '% (image_no, len(self.gt_cells))
        print self.gt_cells
        gt_cells_selected = np.unique(np.array(self.gt_cells))
        if gt_cells_selected.any():
            self.gt_matrix[image_no, gt_cells_selected] = True
        #print self.gt_matrix[image_no, :]

    def save_gt_matrix(self):
        '''Save the gt_matrix to disk'''
        dir_ = self.gt_dir
        if not self.gt_option:
            raise exceptions.IOError('gt_option was not set during the selection of the corners.')
        gt_num = self.gt_option
        gtfile = 'gt_matrix_gt_option%s' %gt_num
        outfile = dir_ + os.sep + gtfile
        np.save(outfile, self.gt_matrix)
        print 'Ground truth self.gt_matrix saved to file %s'% outfile
    
    def load_gt_matrix(self, gt_option):
        '''Load the gt_matrix file and set self.gt_option'''
        dir_ = self.gt_dir
        self.gt_option = gt_option
        gtfile = 'gt_matrix_gt_option%s.npy' %gt_option
        outfile = dir_ + os.sep + gtfile
        gt_matrix = np.load(outfile)
        self.gt_matrix = gt_matrix
        print 'Ground truth self.gt_matrix loaded from file %s'% outfile

    def _analyse_texture(self, data, bins = 16):
        '''Analyse the texture in array data.
        Return vector of six numbers'''
        hist, bin_edges = np.histogram(data, bins, density = False)
        bin_centres = []
        for ii in range(len(bin_edges) - 1):
            centre = np.mean(np.array(bin_edges[ii:ii + 2]))
            bin_centres.append(centre)
        bin_centres = np.array(bin_centres)
        hist = hist / float(np.sum(hist))
        mean = np.sum(bin_centres * hist)
        var = np.sum(np.square(bin_centres - mean) * hist * (1 / ((bins - 1.0) ** 2)))
        sd = np.sqrt(var)
        smoothness = 1 - (1 / (1 + var))
        third_moment = np.sum(np.power(bin_centres - mean, 3) * hist * (1 / ((bins - 1.0) ** 2)))
        uniformity = np.sum(np.square(hist))
        #Replace any zeros with 1e-9
        hist[hist == 0] = 1e-9
        entropy = -1 * np.sum(hist * np.log2(hist))
        texture_vals = np.array((mean, sd, smoothness, third_moment, uniformity, entropy))
        return texture_vals
    
    def _getcell(self, image_data, gridsquare):
        '''Grab the data from gridsquare.
        Return numpy array'''
        try:
            assert gridsquare >= 0
            assert gridsquare < self.ncells_abs
        except:
            print 'gridsquare must be in the range 0, no_cells'
        this_cell = self.grididx[gridsquare]
        #get recorded x & y vals
        x1, y1, x2, y2 = this_cell
        xindxstart = min(x1, x2)
        xindxend = max(x1, x2)
        yindxstart = min(y1, y2)
        yindxend = max(y1, y2)
        data = image_data[yindxstart:yindxend, xindxstart:xindxend]
        return data
    
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
        self.corners.append([event.xdata, event.ydata])

    def _gt_onclick(self, event):
        #print 'Point selected at (%i, %i)'% (round(event.xdata), round(event.ydata)) 
        try:
            cell_selected = self.pixels_to_cellno(event.xdata, event.ydata)
        except exceptions.IOError:
            print "You must click one of the cells."
        else:
            print 'Cell %i selected'% cell_selected
            self.gt_cells.append(cell_selected)        


if __name__ == '__main__':
    S = texture(r'/Users/james/blog/20150918_republicanPCA/cropped_images')
    #S.select_grid_corners()
    S.display_bitdepth_image(0, bins = 256)
    S.display_bitdepth_image(0, bins = 16)
    S.display_bitdepth_image(0, bins = 12)
    #S.groundtruth_select_corners(gt_option = 1)
    #S.create_grid(xpixels = 80, ypixels = 80)
    #S.load_gt_matrix(1)
    #S.display_grid(showarea = False)
    #S.pixels_to_cellno(400, 400)
    #S.measure_all_textures(bins = 16)
    #cell = S._getdata(2)
    #S.plot_raw_textures(image_no = 0, cell_no = None, norm_data = True, centre_data = True, pca_data = True)
    #S.plot_raw_textures(image_no = 0, cell_no = None, norm_data = False, centre_data = True, pca_data = False)
    #S.cluster_cells(image_no = 0, cell_no = None, norm_data = True, method = 'dbscan', kmeans_k = 2 )
    #S.cluster_all_cells(method = 'dbscan', norm_data = True, centre_data = True, pca_data = False, 
    #                    dbscan_eps = 1.5, dbscan_ms = 5)
    #S.plot_clusters_textures(image_no = 0, cell_no = None, pca_data = True)
    #S.plot_clusters_textures(image_no = 0, cell_no = None, norm_data = True, pca_data = False)
    #S.plot_class_textures(image_no = 0, cell_no = None, pca_data = True)
    #S.plot_class_textures(image_no = 0, cell_no = None, norm_data = True, pca_data = False)
#     S.show_anomalies(image_no = 0)
#     S.show_anomalies(image_no = 1)
#     S.show_anomalies(image_no = 2)
#     S.show_anomalies(image_no = 3)
#     S.show_image_cell(0, 0)
#     S.show_image_cell(0, 1)
#     S.show_image_cell(0, 2)
#     S.show_image_cell(0, 3)
#     S.show_image_cell(0, 4)
#     S.show_image_cell(0, 5)
#     S.show_image_cell(0, 6)

    #S.gt_image(0)
#    S.record_all_groundtruth()
    #S.save_gt_matrix()
    
    