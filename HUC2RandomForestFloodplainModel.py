"""
Created on Tuesday November 15 09:25 2016

author:     Jeremy Baynes
contact:    baynes.jeremy@epa.gov

This script creates a random forest (RF) model based on stratified random 
sampling of a single training (Y) and one or more co-registered variable (X) 
dataset(s).  The variable 'stack' can be modified to change the X variables(s).  

This script was written with the following X variables to be used:

HOFD:       Horizontal overland flow distance to channel
OFD:        Overland flow distance to channel
VDC:        Vertical distance to channel
VOFD:       Vertical overland flow distance to channel
nlcd:       National Land Cover Dataset (2011)
hlr:        USGS Hydrologic landscape regions
dem:        National digital elevation model
slope:      Slope derived from dem
cti:        Compound Topographic Index
flood_freq: SSURGO and STATSGO combined flood frequency by soil type
fluvial:    SSURGO and STATSGO combined fluvial soil type

and the following Y variable:

fema:       Fema flood maps reclassified as binary (1) - flood-prone and
            (2) - not flood-prone
            Determined from Fema Flood Hazard Maps and rescaled to 1 if 
            FLD_ZONE = 'A' OR FLD_ZONE = 'A99' OR FLD_ZONE = 'AE' OR 
            FLD_ZONE = 'AH' OR FLD_ZONE = 'AO' OR FLD_ZONE = 'OPEN WATER' 
            OR FLD_ZONE = 'V' OR FLD_ZONE = 'VE'
            else 2
            NoData = 0
            
All X and Y datasets were GeoTiff at 30m resolution and snapped to the same
extent as the 2011 NLCD.

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

def tif_details(img_path):   
    # Returns string with details about tiff including projection and no data
    ds = gdal.Open(img_path)
    rows = ds.RasterYSize
    cols = ds.RasterXSize
    band = ds.GetRasterBand(1)
    datatype = gdal.GetDataTypeName(band.DataType)
    
    prj=ds.GetProjection()
    srs=osr.SpatialReference(wkt=prj)
    if srs.IsProjected:
        projection = '{0} / {1}'.format(srs.GetAttrValue('projcs'), srs.GetAttrValue('geogcs'))
    else:
        projection = srs.GetAttrValue('geogcs')
    string = 'File: {0} loaded\n'.format(img_path)
    string += '{0:<29}Driver:  {1}\n'.format('', ds.GetDriver().LongName)    
    string += '{0:<29}Projection: {1}\n'.format('', projection)
    string += '{0:<29}Geotransform: {1}\n'.format('', ds.GetGeoTransform())
    string += '{0:<29}Rows: {1}\n'.format('', rows)
    string += '{0:<29}Columns: {1}\n'.format('', cols)
    string += '{0:<29}Data type: {1}\n'.format('', datatype)
    string += '{0:<29}NoData value: {1}\n'.format('', band.GetNoDataValue())
    string += '{0:<29}Min: {1}\n'.format('', band.GetMinimum())
    string += '{0:<29}Max: {1}\n'.format('', band.GetMaximum())
    
    ds = None 
    return string
    
def rf_details(rf, variables):
    # Returns string with details about the random forest model
    spacing = 13#len(max(variables, key=len)) + 3
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
                                                      spacing, variable[1].astype('float32'))
       
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
    text_line = timestamp + ' -- ' + text    
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
chunksize = 500
run = 13


indir_base = r'/data/HUCs'
outdir_base = r'/data/rf_output_51_trees_nodata_fix'

stack = [('HOFD', 'numeric'), 
         ('OFD', 'numeric'),
         ('VDC', 'numeric'),
         ('VOFD', 'numeric'),
         ('slope', 'numeric'),
         ('nlcd', 'categorical'),
         ('flood_freq', 'categorical'), ('cti', 'numeric'),
         ('dem_5x5', 'numeric'),  ('fluvial', 'categorical')]

for HUC in range(1,19):
    gc.collect()
    random_seed = 622 * HUC
    basedir = os.path.join(indir_base, 'HUC2_{0:02d}_'.format(HUC))
    outdir = os.path.join(outdir_base, 'HUC_{0}'.format(HUC))
    txtdir = os.path.join(outdir_base, 'Run{0}'.format(run))
    
    # Set output directory and start log file 
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if not os.path.exists(txtdir):
        os.makedirs(txtdir)
    text_file = os.path.join(txtdir, 'Huc_{0}_log_file.txt'.format(HUC))
    stats_file = os.path.join(txtdir, 'Accuracy.txt'.format(HUC))
    if not os.path.exists(stats_file):
        log_write(stats_file, 'HUC, Overall, Precision, Recall, F1_score', 'w')
    
        
    # check if a model has already been created
    existing_model = ([model_pth for model_pth in os.listdir(outdir) 
                       if model_pth.startswith('DRF')])
    
    # If not, create a new model
    if len(existing_model) == 0:
	h2o.init()
        numpy.random.seed(random_seed)
        start_script = datetime.datetime.now()
        log_write(text_file, script_details(), 'w')
        log_write(text_file, 'Random Seed Value: {0}'.format(random_seed))
    
        
        ## Open Training array and random select pixels
        train_path = os.path.join(basedir, 'fema.tif')
        log_write(text_file, 'Open Training (Y) Data')
        log_write(text_file, tif_details(train_path))
        ds = gdal.Open(train_path)
        rows = ds.RasterYSize
        cols = ds.RasterXSize
        print HUC
    
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
        log_write(text_file, 'Opening variable (X) Data')
        
        # Iterate through input variables
        for i, layer in enumerate(stack):
            # Open image path
            img_path = os.path.join(basedir, layer[0] + '.tif')
            log_write(text_file, tif_details(img_path))
            ds = gdal.Open(img_path)
            
            # Create empty array to write EACH variable data to
            train_img = numpy.array([], dtype = 'float32').reshape(0, 1)
            test_img = numpy.array([], dtype = 'float32').reshape(0, 1)
            
            # iterate through random locations and read in values 1 at a time
            # This was done to make the script run on machines with less RAM        
            for pair in training_locations:
                value = ds.GetRasterBand(1).ReadAsArray(int(pair[1]), 
                                                        int(pair[0]), 
                                                        1, 1).astype('float32')
                train_img = numpy.vstack((train_img, value))
            
            # Again but from the testing locations
            for pair in testing_locations:
                value = ds.GetRasterBand(1).ReadAsArray(int(pair[1]), 
                                                        int(pair[0]), 
                                                        1, 1).astype('float32')
                test_img = numpy.vstack((test_img, value))                                     
                                                        
            ds = None
            
            # Append each array of values to a variable training/testing data array
            X_train = numpy.hstack((X_train, train_img)) if X_train.size else train_img
            X_test = numpy.hstack((X_test, test_img)) if X_test.size else test_img
           
        log_write(text_file, 'All input variables loaded')
    
        
        # Random Forest model with h2o
        col_names = [layer[0] for layer in stack] + ['fema']
        col_types = [layer[1] for layer in stack] + ['categorical']
                 
        #load data to h2o frame    
        data = h2o.H2OFrame(numpy.hstack((X_train, Y_train[:,None])), column_types=col_types)
        data.col_names = col_names
        
        # model parameters
        h2o_rf_model = (H2ORandomForestEstimator(balance_classes = False, 
                                                 ntrees = 51, mtries=4, 
                                                 max_depth=12, 
                                                 seed=random_seed,
                                                 score_each_iteration = True))
        x_variables = [layer[0] for layer in stack]
        y_variable = 'fema'
        
        # Train RF model
        h2o_rf_model.train(x_variables, y_variable, data)
        log_write(text_file, rf_details(h2o_rf_model, stack))
        
        # Save model to file
        rf_model_file = h2o.save_model(h2o_rf_model, outdir, True)
        log_write(text_file, 'Random Forest Model stored: {0}'.format(rf_model_file))
       
        # load testing data
        test_data = h2o.H2OFrame(X_test, column_types=col_types[:-1])
        test_data.col_names = col_names[:-1]
        
        # Predict testing data and create confusion matrix
        Y_pred = h2o_rf_model.predict(test_data)
        Y_pred = (numpy.array(h2o.as_list(Y_pred, use_pandas=False, 
                  header=False))[:,0].astype('float32').astype('uint8'))
        cm = ConfusionMatrix(Y_test, Y_pred) 
        classification_report = cm.classification_report
        cm_stats = ('{0}, {1}, {2}, {3}, {4}'.format(HUC,
                                     cm.ACC,
                                     classification_report.iloc[0,0],
                                     classification_report.iloc[0,1],
                                     classification_report.iloc[0,2]))
        log_write(stats_file, cm_stats)
        log_write(text_file, aa_details(cm))
    
        # Clean Variables
        X_test = None
        X_train = None
        Y_test = None
        Y_pred = None
        Y_train = None
        test_img = None
        train_img = None
    
    # Else use existing model
    else:
        try:
            h2o.init()
            h2o_rf_model = h2o.load_model(os.path.join(outdir, existing_model[0]))
            
        except:
            time.sleep(20)
            h2o.init()
            h2o_rf_model = h2o.load_model(os.path.join(outdir, existing_model[0]))
            
        
        train_path = os.path.join(basedir, 'fema.tif')
        ds = gdal.Open(train_path)
        rows = ds.RasterYSize
        cols = ds.RasterXSize
    
    ##########################################################################
    ######### Model Created or Loaded --  Classify all input data ############
    ##########################################################################
    
    os.system('rm -rf /tmp/*')
    os.system('rm -rf /var/tmp/*')
    
    progress_file = os.path.join(outdir, 'progress.txt')
    if os.path.exists(progress_file):
        readfile = open(progress_file, 'r')
        line = readfile.readline()
        chunk = int(line.split(':')[1])
        readfile.close()
	log_write(text_file, 'Script restarted')
        
        
    else:
        chunk = 0
        # Create empty array same size as input arrays and save as .tif
        classification = numpy.zeros((rows, cols)).astype('uint8')
        output = os.path.join(outdir, 'RF_Output_{0:02d}.tif'.format(HUC))
        write_tiff(classification, train_path, output)
        
        probability = numpy.zeros((rows, cols)).astype('uint8')
        output = os.path.join(outdir, 'RF_Probability_{0:02d}.tif'.format(HUC))
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
            
	    try:
                h2o_X = h2o.H2OFrame(X, column_types=col_types[:-1])
                h2o_X.col_names = col_names[:-1]
                
                chunk_prediction = rf_prediction(h2o_rf_model, h2o_X)
		   
            except: 
                # would run into memory problems. shutting down and 
                # restarting seemed to fix
                h2o.remove_all()
                h2o.cluster().shutdown()
                time.sleep(30)
                try:
                    h2o.init()
                except:
                    time.sleep(20)
                    h2o.init()
                h2o_X = h2o.H2OFrame(X, column_types=col_types[:-1])
                h2o_X.col_names = col_names[:-1]
                h2o_rf_model = h2o.load_model(os.path.join(outdir, [model_pth for model_pth in 
                                       os.listdir(outdir) if 
                                       model_pth.startswith('DRF')][0]))
                chunk_prediction = rf_prediction(h2o_rf_model, h2o_X)
                h2o_X = None
           
            # Reshape back to original array shape
            chunk_prediction_array = (numpy.array(h2o.as_list(chunk_prediction, 
                                      use_pandas=False, header=False))[:,0]
                                     .astype('float32').astype('uint8'))
            #X = numpy.zeros(rows*x_size).astype('uint8')
            #X[valid_data_locations] = chunk_prediction_array
            X = chunk_prediction_array.reshape((rows, x_size))
            
            # Save prediction to classification array
            output = os.path.join(outdir, 'RF_Output_{0:02d}.tif'.format(HUC))
            write_tiff_chunk(X, output, (x_offset, y_offset, x_size, y_size))
            
            
            # same for probability 
            chunk_probability_array = (numpy.array(h2o.as_list(chunk_prediction, 
                                       use_pandas=False, header=False))[:,1]
                                      .astype('float32') * 100)
            #X = numpy.zeros(rows*x_size).astype('uint8')
            #X[valid_data_locations] = chunk_probability_array
            
            X = chunk_probability_array.reshape((rows, x_size))
            output = os.path.join(outdir, 'RF_Probability_{0:02d}.tif'.format(HUC))
            write_tiff_chunk(X, output, (x_offset, y_offset, x_size, y_size))
    
            
            # Write to log file
            if chunk == 0:
                string = 'Begin flood classification from input variables\n'
                string += '{0:<29}{1:.<{2}} {3}\n'.format('', 'Number of chunks', 19, number_of_chunks)
                string += '{0:<29}{1:.<{2}} {3}\n'.format('', 'Chunk size', 19, chunksize)
                log_write(text_file, string)
            
            if chunk + 1 == number_of_chunks:
                log_write(text_file, 'RF classification of input variables completed')
    
    
            writefile = open(progress_file, 'w')
            writefile.write('chunk:{0}'.format(chunk+1))
            writefile.close()
            chunk += 1
        
        if not os.path.exists('RF_Output_{0:02d}.ovr'.format(HUC)):
            try:
                output = os.path.join(outdir, 'RF_Output_{0:02d}.tif'.format(HUC))
                command = 'gdaladdo -ro "{0}" 2 4 8 16 32 64'.format(output)
                os.system(command)
                log_write(text_file, 'Classification output: {0}'.format(output))
                
                # Build pyramids
                output = os.path.join(outdir, 'RF_Probability_{0:02d}.tif'.format(HUC))
                command = 'gdaladdo -ro "{0}" 2 4 8 16 32 64'.format(output)
                os.system(command)
                log_write(text_file, 'Probability output: {0}'.format(output))
            except Exception, e:
                log_write(text_file, '**Error** Pyrmaids failed')
                log_write(text_file, e)
       
        log_write(text_file, 'Script completed')

    
 
