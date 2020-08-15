import scipy.misc
import glob
import imageio
##img_array = scipy.misc.imread(image_for_demo.png,flatten = True)
##
##img_data = 255.0 img_array . reshape( 784 )
##img_data = (img_data / 255.0 * 0.99 ) + 0.01


import numpy
# scipy.special for the sigmoid function expit()
import scipy.special
# library for plotting arrays
import matplotlib.pyplot

# our own image test data set
our_own_dataset = []


from skimage import transform,io
# read in grey-scale
grey = io.imread('2828_my_own_1.png', as_grey=True)
# resize to 28x28
small_grey = transform.resize(grey, (28,28), mode='symmetric', preserve_range=True)
# reshape to (1,784)
reshape_img = small_grey.reshape(1, 784)

"""
# load the png image data as test data set
for image_file_name in glob.glob('2828_my_own_1.png'):
    
    # use the filename to set the correct label
    #label = int(image_file_name[-5:-4])
    
    # load image data from png files into an array
    print ("loading ... ", image_file_name)
    img_array = imageio.imread(image_file_name, as_gray=True)
    
    # reshape from 28x28 to list of 784 values, invert values
    img_data  = 255.0 - img_array.reshape(784)
    
    # then scale data to range from 0.01 to 1.0
    img_data = (img_data / 255.0 * 0.99) + 0.01
    print(numpy.min(img_data))
    print(numpy.max(img_data))
    
    # append label and image data  to test data set
    record = numpy.append(img_data)
    our_own_dataset.append(record)
    
    pass

"""
