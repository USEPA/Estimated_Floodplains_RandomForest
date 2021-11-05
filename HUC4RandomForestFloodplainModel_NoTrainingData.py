"""
Created on Tuesday November 15 09:25 2016

author:	 Jeremy Baynes
contact:	baynes.jeremy@epa.gov

This script uses one or more existing RF model(s) to classify floodplain
Developed for (2) 4-digit HUCs that had no training data
The 4-digit HUCs that bounded were used

"""

# import all required Python packages:
import os
import sys
import socket
import platform
import psutil
import datetime
import pytz
import time
import gc

import numpy
import gdal
import osr

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
	# Writes an array as Tiff with the same properties as an input Tiff
	# Useful for writing the processed results of a Tiff with the same 
	# Spatial parameters (extent, projection, cell size, etc.)
	  
	ds = gdal.Open(Gtiff, gdal.GA_Update)
	ds.GetRasterBand(1).WriteArray(A, offsets[0], offsets[1])
	ds = None
	
	return None
	
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

def rf_prediction(model, data):
	prediction = model.predict(data)
	return prediction
	
###########################--Input Parameters--###########################

# Set number of random training pixels to create the RF Model
samples = 500000
chunksize = 2000
run = 16


indir_base = r'/data/HUC4s'
outdir_base = r'/data/HUC4_80_20_Split'
txtdir = os.path.join(outdir_base, 'Run{0}'.format(run))


if not os.path.exists(txtdir):
	os.makedirs(txtdir)

stats_file = os.path.join(txtdir, 'Accuracy.txt')
if not os.path.exists(stats_file):
	log_write(stats_file, 'HUC, Overall, Precision, Recall, F1_score', 'w')
error_file = os.path.join(txtdir, 'Errors.txt')
if not os.path.exists(error_file):
	log_write(error_file, 'HUC, Error', 'w')


stack = [('HOFD', 'numeric'), 
		 ('OFD', 'numeric'),
		 ('VDC', 'numeric'),
		 ('VOFD', 'numeric'),
		 ('slope', 'numeric'),
		 ('nlcd', 'categorical'),
		 ('flood_freq', 'categorical'), ('cti', 'numeric'),
		 ('dem_5x5', 'numeric'),  ('fluvial', 'categorical')]

huc4_ids = [904]
use_models = [1003, 1005, 1701]

#huc4_ids = [903]
#use_models = [401, 701, 902]

for HUC in huc4_ids:
	gc.collect()
	h2o.init()
	random_seed = 622 * HUC
	basedir = os.path.join(indir_base, 'HUC4_{0:04d}'.format(HUC))
	outdir = os.path.join(outdir_base, 'HUC4_{0:04d}'.format(HUC))
	
	
	# Set output directory and start log file 
	if not os.path.exists(outdir):
		os.makedirs(outdir)
	text_file = os.path.join(txtdir, 'Huc_{0:04d}_log_file.txt'.format(HUC))

	## Open Training array and random select pixels
	train_path = os.path.join(basedir, 'fema.tif')
	ds = gdal.Open(train_path)
	rows = ds.RasterYSize
	cols = ds.RasterXSize

	
	
	# # Else use existing model
	for HUC_model in use_models:
		model_pth = os.path.join(outdir_base, 'HUC4_{0:04d}'.format(HUC_model))
		
		existing_model = ([m for m in os.listdir(model_pth) if m.endswith('DRF')])
		print model_pth, existing_model
		h2o_rf_model = h2o.load_model(os.path.join(model_pth, existing_model[0]))
	
	
		chunk = 0
		# Create empty array same size as input arrays and save as .tif
		classification = numpy.zeros((rows, cols)).astype('uint8')
		output = os.path.join(outdir, 'RF_Output_{0:04d}.tif'.format(HUC_model))
		write_tiff(classification, train_path, output)
		
		probability = numpy.zeros((rows, cols)).astype('uint8')
		output = os.path.join(outdir, 'RF_Probability_{0:04d}.tif'.format(HUC_model))
		write_tiff(probability, train_path, output)
		
	
		# Divide the input arrays into chunks of n columns by all rows
		number_of_chunks = cols/chunksize + 1 if cols%chunksize else cols/chunksize
		if chunk < number_of_chunks:
			
			while chunk < number_of_chunks:
			   
				print '{0} / {1}'.format(chunk + 1, number_of_chunks)
				
				# Chunksize variables
				x_offset = chunk * chunksize
				y_offset = 0
				x_size = min(cols - x_offset, chunksize)  # accounts for last chunk 
				y_size = rows
				
				# Open each dataset using chunksize variables
				for i, layer in enumerate([datalayer[0] for datalayer in stack]):
					img_path = os.path.join(basedir, layer + '.tif')
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
				
				# no need to predict nodata values
				#valid_data_locatIONS = Numpy.where(numpy.isfinite(numpy.sum(X,1)))[0]
				#X = X.flatten()#[valid_data_locatioNS]
				
				# Predict with RF model
				col_names = [layer[0] for layer in stack] + ['fema']
				col_types = [layer[1] for layer in stack] + ['categorical']

			
	
				h2o_X = h2o.H2OFrame(X, column_types=col_types[:-1])
				h2o_X.col_names = col_names[:-1]
				
				chunk_prediction = rf_prediction(h2o_rf_model, h2o_X)
		   
	#
		   
				# Reshape back to original array shape
				chunk_prediction_array = (numpy.array(h2o.as_list(chunk_prediction, 
										  use_pandas=False, header=False))[:,0]
										 .astype('float32').astype('uint8'))
				#X = numpy.zeros(rows*x_size).astype('uint8')
				#X[valid_data_locations] = chunk_prediction_array
				X = chunk_prediction_array.reshape((rows, x_size))
				
				# Save prediction to classification array
				output = os.path.join(outdir, 'RF_Output_{0:04d}.tif'.format(HUC_model))
				write_tiff_chunk(X, output, (x_offset, y_offset, x_size, y_size))
				
				
				# same for probability 
				chunk_probability_array = (numpy.array(h2o.as_list(chunk_prediction, 
										   use_pandas=False, header=False))[:,1]
										  .astype('float32') * 100)
				#X = numpy.zeros(rows*x_size).astype('uint8')
				#X[valid_data_locations] = chunk_probability_array
				
				X = chunk_probability_array.reshape((rows, x_size))
				output = os.path.join(outdir, 'RF_Probability_{0:04d}.tif'.format(HUC_model))
				write_tiff_chunk(X, output, (x_offset, y_offset, x_size, y_size))

				chunk += 1
	
				
		
log_write(text_file, 'Script completed')

	
 
