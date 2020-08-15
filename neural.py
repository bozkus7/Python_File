import numpy
import matplotlib.pyplot



data_file = open('mnist_train_100.csv.txt', 'r')
data_list = data_file.readlines()
data_file.close()


#The numpy.asfarray() is a numpy function to convert the text strings into real numbers and to create an array of those numbers


all_values = data_list[0].split(',')
image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
print(matplotlib.pyplot.imshow(image_array, cmap = 'Greys', interpolation = 'None'))

"""
Dividing the raw inputs which are in the range 0255
by 255 will bring them into the range 01.
We then need to multiply by 0.99 to bring them into the range 0.0 0.99.
We then add 0.01 to
shift them up to the desired range 0.01 to 1.00. The following Python code shows this in action
"""

scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
print(scaled_input)


#output node is 10 (ex)
onodes = 10
targets = numpy.zeros(onodes) + 0.01
targets[int(all_values[0])] = 0.99

