'''
Created on 19 Dec 2015

@author: james
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def onclick(event):
    print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
        event.button, event.x, event.y, event.xdata, event.ydata)
    print '(%i, %i)'% (round(event.xdata), round(event.ydata)) 

def read_jpg(filename = '/Users/james/workspace/texture_analysis/DT.jpg'):
    '''Read jpg file into numpy array'''
    img = mpimg.imread(filename)
    img = np.mean(img, axis = 2)
    #plt.imshow(img, cmap = 'hot')
    #plt.show()
    return img

def select_points(data):
    '''Select points on image'''
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    ax.imshow(data, cmap = 'hot')
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    fig.canvas.mpl_disconnect(cid)
        
if __name__ == '__main__':
    data = read_jpg()
    select_points(data)