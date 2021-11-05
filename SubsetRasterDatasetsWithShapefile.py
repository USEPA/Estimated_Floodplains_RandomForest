"""
author:     Jeremy Baynes
contact:    baynes.jeremy@epa.gov

This script takes a set of raster datasets and a shapefile to 
extract each dataset to the same extent (i.e., same columns and rows).

Primarily used to subset a series of large rasters (e.g., national extent) with 
a vector datasets (e.g., HUCs, counties, states, etc)


"""

import os
import collections

import arcpy

# shapefile with features that will be used to clip rasters
clip_shp = r"D:\HUC_boundaries\EnviroAtlas_HUC4.shp"
fieldname = 'HUC4'

# output raster size in meters
raster_size = 30

# input/output path locations
input_pth = r'D:\Fema_Project_Layers\National'
output_pth = r'C:\Users\jbaynes\Desktop\HUC4s'
scratch_pth = r'C:\Users\JBaynes\Desktop\temp\junk'

# National rasters    
# data consists of tuple (output_name, filepath)    
bands = (('fema', os.path.join(input_pth, "fema_flood_binary_20170313.tif")),
         ('HOFD', os.path.join(input_pth, "HOFD_nodata.tif")),
         ('VDC', os.path.join(input_pth, "VDC_nodata.tif")),
         ('OFD', os.path.join(input_pth, "OFD_nodata.tif")),
         ('VOFD', os.path.join(input_pth, "VOFD_nodata.tif")),
         ('slope', os.path.join(input_pth, "US_Slope.tif")),
         ('cti', os.path.join(input_pth, "US_CTI.tif")),
         ('dem_5x5', os.path.join(input_pth, "dem_5x5.tif")),
         ('nlcd', os.path.join(input_pth, "2011_Level1_NLCD.tif")),
         ('flood_freq', os.path.join(input_pth, "fld_dcd_final.tif")),
         ('fluvial', os.path.join(input_pth, 'fluvclass.tif')),

         #('hlr', os.path.join(input_pth, "HLR_raster.tif")), Dropped
         #('plan', os.path.join(input_pth, "US_plan.tif")), Dropped
         #('profile', os.path.join(input_pth, "US_profile.tif")), Dropped
         #('curvature', os.path.join(input_pth, "US_curvature.tif")), Dropped
         #('dem', os.path.join(input_pth, "US_DEM.tif")),  Dropped
         #('dem_3x3', os.path.join(input_pth, "dem_3x3.tif")),  Dropped
         )
stack = collections.OrderedDict(bands)

# arcpy parameters
arcpy.env.workspace = scratch_pth
arcpy.CheckOutExtension('SPATIAL')
arcpy.env.overwriteOutput = 1
arcpy.env.snapRaster = os.path.join(input_pth, "2011_Level1_NLCD.tif")
arcpy.env.compression = 'LZW'
arcpy.env.pyramid = None

# iterate features in shapefile to build list of unique ids
uniqueIDs = []
rows = arcpy.SearchCursor(clip_shp)
for row in rows:
    uniqueIDs.append(row.getValue(fieldname))
row = None
rows = None

# Create the boundary rasters first then use that as the extent 
for i in uniqueIDs:  
    
    # Set output directory
    outdir = os.path.join(output_pth, '{0}_{1}'.format(fieldname, i))
    if not os.path.exists(outdir):
        os.makedirs(outdir)    
        
    # Copy feature to new shapefile
    arcpy.MakeFeatureLayer_management(clip_shp, 'temp_layer', """{0} = 
                                      '{1}'""".format(fieldname, i))
    arcpy.CopyFeatures_management('temp_layer',  'temp_feature.shp')
    

    # convert feature vector to raster boundary
    boundary_pth = os.path.join(outdir, 'boundary.tif')
    temp_tif = 'temp_feature.tif'
    arcpy.FeatureToRaster_conversion('temp_feature.shp', 
                                     fieldname, 
                                     temp_tif, 
                                     raster_size)
    
    arcpy.CopyRaster_management(temp_tif, boundary_pth)    
    arcpy.Delete_management(temp_tif)


for i in uniqueIDs:
    for raster in stack:
        temp_raster = 'temp_raster'
        
        outdir = os.path.join(output_pth, '{0}_{1}'.format(fieldname, i))
        boundary_pth = os.path.join(outdir, 'boundary.tif')
        arcpy.env.extent = arcpy.Raster(boundary_pth)
        arcpy.MakeRasterLayer_management(boundary_pth, temp_raster)
        clip_raster_pth = os.path.join(outdir, '{0}.tif'.format(raster, i))  
         
        # Clip 
        clip_raster = arcpy.sa.ExtractByMask(stack[raster], temp_raster)
        if os.path.exists(clip_raster_pth):
            arcpy.Delete_management(clip_raster_pth)
        arcpy.CopyRaster_management(clip_raster, clip_raster_pth)
        
        
        print i, raster   
        arcpy.Delete_management(temp_raster)

    print '#' * 50 + '\n'
#




