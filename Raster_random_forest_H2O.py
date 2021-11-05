"""
Created on Tuesday November 15 09:25 2016

author:	 Jeremy Baynes
contact:	baynes.jeremy@epa.gov

This script creates a random forest (RF) model and classification
based on random sampling of a single training (Y) and one or more 
co-registered explanatory variable (X) dataset(s).  The variable 
'stack' can be modified to change the X variables(s).  


"""

# import packages:
import os
import sys
import socket
import platform
import psutil
import datetime
import pytz
import time

import numpy
import gdal

from pandas_ml import ConfusionMatrix

import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator

def write_tiff(A, Gtiff, output):
	# Writes an array as Tiff with the same properties as an input Tiff
	# Useful for writing the processed results of a Tiff with the same 
	# Spatial parameters (extent, projection, cell size, etc.)
	cols = A.shape[1]
	rows = A.shape[0]   
	ds = gdal.Open(Gtiff)
	driver = gdal.GetDriverByName('GTiff')
	outRaster = driver.Create(output, cols, rows, 1, gdal.GDT_Byte)
	outRaster.SetGeoTransform(ds.GetGeoTransform())
	outRaster.SetProjection(ds.GetProjection())
	outband = outRaster.GetRasterBand(1)
	outband.WriteArray(A)
	outband = None
	
	return None
	
def write_tiff_chunk(A, Gtiff, offsets):
	# Writes an array as a subset of a Tiff	  
	ds = gdal.Open(Gtiff, gdal.GA_Update)
	ds.GetRasterBand(1).WriteArray(A, offsets[0], offsets[1])
	ds = None
	
	return None

def script_details():
	# Returns string with details about script/user/workstation
	current_time = datetime.datetime.now(pytz.timezone('US/Eastern'))

	string = 'Script started'
	string += '\n' + '*'*50 + '\n'
	string += "Workstation Name: {0}\n".format(socket.gethostname())
	string += "FQDN: {0}\n".format(socket.getfqdn())
	string += "System OS: {0}\n".format(platform.system())
	string += "Platform: {0}\n".format(platform.platform())
	string += "Pocessor: {0}\n".format(platform.processor())
	string += "Release: {0}\n".format(platform.release())
	string += "Number of CPUs: {0}\n".format(psutil.cpu_count())
	cores = psutil.cpu_count(logical=False)
	string += "Number of Physical CPUs: {0}\n".format(cores)
	memory = psutil.virtual_memory().total
	string += "Ram: {0} GB\n".format(int(round(memory/(2**30.0))))
	string += "Python: {0}\n".format(platform.python_version())
	string += 'Script Path: {0}\n'.format(sys.argv[0])
	string += 'User: {0}\n'.format(os.getenv('USERNAME'))
	string += 'Run Date: {0}\n'.format(current_time.strftime("%B %m, %Y"))
	string += 'Run Time: {0}\n'.format(current_time.strftime("%H:%M:%S %Z"))
	string += '*'*50 + '\n\n'
	
	return string

	
def rf_details(rf, variables):
	# Returns string with details about the random forest model
	importances = numpy.array(rf.varimp())[:,[0,3]]
	#indices = numpy.argsort(importances)[::-1]
	string = 'Random Forest Model created\n'
	string += '{0:<29}Random training samples: {1}\n\n'.format('', len(training_locations))
	string += '{0:<29}{1}\n'.format('', '***Random Forest Parameters***')
	string += '{0:<29}{1:.<{2}} {3}\n'.format('', 'Number of classes', 22, 2)#rf.n_classes_)
	string += '{0:<29}{1:.<{2}} {3}\n'.format('', 'Number of variables', 22, len(variables))
	string += '{0:<29}{1:.<{2}} {3}\n'.format('', 'Number of trees', 22, rf.ntrees)
	string += '{0:<29}{1:.<{2}} {3}\n'.format('', 'OOB Score', 22, rf.accuracy()[0][1])
	string += '\n{0:<29}***Feature Importance***\n'.format('')
	for variable in importances:
		string += '{0:<29}{1:.<{2}} {3:.5f}\n'.format('',variable[0], 
													  22, variable[1].astype('float32'))
	   
	return string

def aa_details(cm):
	# Returns string with details about random forest accuracy assessment
	string = '\n' + '*'*50 + '\n'
	string += 'Confusion matrix and stats of completely random points\n'
	string += str(cm)
	string += '\n\n'
	try:
		string += str(cm.classification_report)
	except:
		string += 'Error printing confusion statistics'
	string += '\n' + '*'*50 + '\n\n'
	return string


	
def log_write(file_path, text, mode = 'a'):
	# Writes string with a timestamp to a given text file
	current_time = datetime.datetime.now(pytz.timezone('US/Eastern'))
	timestamp = current_time.strftime("[%Y%m%d - %H:%M:%S %Z]")
	text_line = timestamp + ', ' + text	
	txt_file = open(file_path, mode)
	txt_file.write(text_line)
	txt_file.write('\n')	 
	txt_file.close()
	print text_line

	return None

	
###########################--Input Parameters--###########################

# input and output directories
basedir = r'/data/share'
outdir = r'/data/share/output'

# Set number of random training pixels to create the RF Model
samples = 500000

# maximum number of columns to classify at a time (RAM limited)
# the more explanatory variables the smaller this will need to be
chunksize = 1000

# set a random seed value
random_seed = 622

# log file
text_file = os.path.join(outdir, 'log_file.txt')

## the file names and data type of the explanatory variables.  These
## need to be in the basedir directory.  All of these and the 
## response variable should be the same size (i.e., cols and rows) 
## and extent
stack = [('HOFD.tif', 'numeric'), 
		 ('OFD.tif', 'numeric'),
		 ('VDC.tif', 'numeric'),
		 ('VOFD.tif', 'numeric'),
		 ('slope.tif', 'numeric'),
		 ('nlcd.tif', 'categorical'),
		 ('flood_freq.tif', 'categorical'), ('cti.tif', 'numeric'),
		 ('dem_5x5.tif', 'numeric'),  ('fluvial.tif', 'categorical')]

# set response variable
rv = ('fema.tif', 'categorical')
rv_valid_values = [1, 2]


# classification and probability output
rf_classification_output = os.path.join(outdir, 'new_classification.tif')
rf_probability_output = os.path.join(outdir, 'new_probability.tif')

#rf-model hyper-parameters
model_id = 'new_drf'
balance_classes = True
ntrees = 51
mtries=4
max_depth=12


###########################--Input Parameters--###########################


if not os.path.exists(outdir):
	os.makedirs(outdir)
   

h2o.init()
numpy.random.seed(random_seed)
start_script = datetime.datetime.now()
log_write(text_file, script_details(), 'w')
log_write(text_file, 'Random Seed Value: {0}'.format(random_seed))


## Open response variable array and random select pixels
train_path = os.path.join(basedir, rv[0])
ds = gdal.Open(train_path)
rows = ds.RasterYSize
cols = ds.RasterXSize


train_array = ds.GetRasterBand(1).ReadAsArray()

# Randomly select n pixels each from classes 1 and 2
# Return 2d index value of each location that is equal to valid_train_range
train_population = numpy.asarray(numpy.where(train_array > 0), dtype= 'uint32').T

# Randomly select n index values from train_population
random_train = numpy.random.choice(numpy.arange(len(train_population)), 
                                       samples, replace = False)  

# Array of stratified random locations
training_locations = train_population[random_train]

# Now create array of random locations for testing
train_population = None
train_population = numpy.asarray(numpy.where(train_array > 0), dtype= 'uint32').T

random_train = numpy.random.choice(numpy.arange(len(train_population)), 
                                           samples, replace = False)
testing_locations = train_population[random_train]

# Get test and training values from random_locations
Y_train = train_array[training_locations[:,0], training_locations[:,1]]
Y_test = train_array[testing_locations[:,0], testing_locations[:,1]]



# Clear variables before moving on
ds = None
train_array = None
train_population = None
random_train = None

# need to do this because not enough RAM to load all variables at once
# Start with empty array to hold data from ALL input variables
X_train = numpy.array([], dtype = 'float32')
X_test = numpy.array([], dtype = 'float32')
# Get variable values from random_location

# Iterate through input variables
for i, layer in enumerate(stack):
	# Open image path
	img_path = os.path.join(basedir, layer[0])
	ds = gdal.Open(img_path)
	
	value = ds.GetRasterBand(1).ReadAsArray()
	train_img = value[training_locations[:,0], training_locations[:,1]]
	test_img = value[testing_locations[:,0], testing_locations[:,1]]

		
	# Append each array of values to a variable training/testing data array
	X_train = numpy.dstack((X_train, train_img)) if X_train.size else train_img
	X_test = numpy.dstack((X_test, test_img)) if X_test.size else test_img
	   

# add response variable to column names and types
col_names = [layer[0] for layer in stack] + [rv[0]]
col_types = [layer[1] for layer in stack] + [rv[1]]
 
#load data to h2o frame	
data = h2o.H2OFrame(numpy.hstack((X_train[0], Y_train[:,None])), column_types=col_types)
data.col_names = col_names

# model parameters
h2o_rf_model = (H2ORandomForestEstimator(balance_classes = balance_classes, 
										 ntrees = ntrees, mtries=mtries, 
										 max_depth=max_depth, 
										 seed=random_seed,
										 score_each_iteration = True,
										 model_id = model_id))
x_variables = [layer[0] for layer in stack]
y_variable = rv[0]

# Train RF model
h2o_rf_model.train(x_variables, y_variable, data)
log_write(text_file, rf_details(h2o_rf_model, stack))

# Save model to file
rf_model_file = h2o.save_model(h2o_rf_model, outdir, True)
log_write(text_file, 'Random Forest Model stored: {0}'.format(rf_model_file))

# load testing data
test_data = h2o.H2OFrame(X_test[0], column_types=col_types[:-1])
test_data.col_names = col_names[:-1]

# Predict testing data and create confusion matrix
Y_pred = h2o_rf_model.predict(test_data)
Y_pred = (numpy.array(h2o.as_list(Y_pred, use_pandas=False, 
		  header=False))[:,0].astype('float32').astype('uint8'))

cm = ConfusionMatrix(Y_test, Y_pred) 
classification_report = cm.classification_report
log_write(text_file, 'Accuracy statistics\n' + classification_report.to_string())


# Clean Variables
X_test = None
X_train = None
Y_test = None
Y_pred = None
Y_train = None
test_img = None
train_img = None



##########################################################################
############### Model Created --  Classify all input data ################
##########################################################################


# Create empty array same size as input arrays and save as .tif
classification = numpy.zeros((rows, cols)).astype('uint8')
write_tiff(classification, train_path, rf_classification_output)

probability = numpy.zeros((rows, cols)).astype('uint8')
write_tiff(probability, train_path, rf_probability_output)

# classify in chunks (RAM limited to load entire datasets at once)
number_of_chunks = cols/chunksize + 1 if cols%chunksize else cols/chunksize
chunk = 0
while chunk < number_of_chunks:

	# Write to log file
	if chunk == 0:
		string = 'Classification from input variables in chunks\n'
		string += '{0:<29}{1:.<{2}} {3}\n'.format('', 'Number of chunks', 19, number_of_chunks)
		string += '{0:<29}{1:.<{2}} {3} columns\n'.format('', 'Chunk size', 19, chunksize)
		log_write(text_file, string)

	# Chunksize variables
	x_offset = chunk * chunksize
	y_offset = 0
	x_size = min(cols - x_offset, chunksize)  # accounts for last chunk 
	y_size = rows

	# Open each dataset using chunksize variables
	for i, layer in enumerate([datalayer[0] for datalayer in stack]):
		img_path = os.path.join(basedir, layer)
		ds = gdal.Open(img_path)
		if ds.RasterYSize < y_size:
			y_size = ds.RasterYSize
		if ds.RasterXSize < cols:
			x_size = min(ds.RasterXSize - x_offset, chunksize)
		
		# Read input tif in with chunk size parameters
		value = ds.GetRasterBand(1).ReadAsArray(x_offset, y_offset, 
												x_size, y_size).astype('float32')
		# Convert from 2d to 1d
		new_shape = (value.shape[0] * value.shape[1], 1) 
		value = value.reshape(new_shape)

		# Build 2d array where each column is data from one input tiff 
		if i == 0:
			X = value
		else:
			X = numpy.hstack((X, value))

	
	# input data loaded to X, now predict

	# Predict with RF model
	col_names = [layer[0] for layer in stack] + ['fema']
	col_types = [layer[1] for layer in stack] + ['categorical']


	h2o_X = h2o.H2OFrame(X, column_types=col_types[:-1])
	h2o_X.col_names = col_names[:-1]
	chunk_prediction = h2o_rf_model.predict(h2o_X)
	
	# Reshape back to original array shape
	chunk_prediction_array = (numpy.array(h2o.as_list(chunk_prediction, 
							  use_pandas=False, header=False))[:,0]
							 .astype('float32').astype('uint8'))

	X = chunk_prediction_array.reshape((rows, x_size))
	# Save prediction to classification array
	write_tiff_chunk(X, rf_classification_output, (x_offset, y_offset, x_size, y_size))


	# same for probability 
	chunk_probability_array = (numpy.array(h2o.as_list(chunk_prediction, 
							   use_pandas=False, header=False))[:,1]
							  .astype('float32') * 100)

	X = chunk_probability_array.reshape((rows, x_size))
	write_tiff_chunk(X, rf_probability_output, (x_offset, y_offset, x_size, y_size))


	# finished chunk, write to log file
	string = 'Completed chunk: {0} / {1}'.format(chunk + 1, number_of_chunks)
	log_write(text_file, string)


	chunk += 1

#	#Build pyramids for output and probability
# 	command = 'gdaladdo -ro "{0}" 2 4 8 16 32 64'.format(rf_classification_output)
# 	os.system(command)
# 	log_write(text_file, 'Classification output: {0}'.format(rf_classification_output))
	

# 	command = 'gdaladdo -ro "{0}" 2 4 8 16 32 64'.format(rf_probability_output)
# 	os.system(command)
# 	log_write(text_file, 'Probability output: {0}'.format(rf_probability_output))

log_write(text_file, 'Script completed')
	
	
