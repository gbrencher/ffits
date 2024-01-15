#----------------------------------------------------------
# isce topsStack
# Import required packages
import os
import sys
import logging
import isce
root_logger = logging.getLogger()
root_logger.setLevel('WARNING')
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from datetime import datetime, timedelta
import time
from glob import glob
import asf_search as asf
import subprocess
import yaml

with open("config.yml", 'r') as stream:
    args = yaml.safe_load(stream)

# Set environment variables to call ISCE from the command line
os.environ['ISCE_HOME'] = os.path.dirname(isce.__file__)
os.environ['ISCE_ROOT'] = os.path.dirname(os.environ['ISCE_HOME'])
os.environ['ISCE_STACK'] = '/mnt/Backups/gbrench/sw/insar_tools/isce2/src/isce2/contrib/stack' # probably need to edit
# can potentially use https://github.com/ASFHyP3/hyp3-isce2/blob/develop/src/hyp3_isce2/__init__.py
os.environ['PYTHONPATH'] = os.environ['ISCE_STACK']
os.environ['PATH'] += f":{os.environ['ISCE_STACK']}/topsStack"

# set local processing path
proc_path = './data'
os.chdir(proc_path)

# initialize directories
os.makedirs(f'{proc_path}/slc', exist_ok=True)
os.makedirs(f'{proc_path}/orbits', exist_ok=True)
os.makedirs(f'{proc_path}/aux', exist_ok=True)
os.makedirs(f'{proc_path}/dem', exist_ok=True)

# download slcs
results = asf.granule_search(args['scene_list'])
results.download(path=f'{proc_path}/slc', processes=2)

# download dem (padded by 1 degree around aoi, assumes northern hemisphere currently)
os.chdir(f'{proc_path}/dem')
subprocess.run([f"sardem --bbox {args['bbox']['E']-1} {args['bbox']['S']-1} {args['bbox']['W']+1} {args['bbox']['N']+1} --xrate 3 --yrate 3 --data-source COP -isce "], shell=True, capture_output=True, text=True) # will need to make bbox an input
#!sardem --bbox 85.9 27.1 87.6 28.6 --xrate 3 --yrate 3 --data-source COP -isce 

# stacksentinel to generate run files for cslc workflow
subprocess.run([f"stackSentinel.py -s {proc_path}/slc -o {proc_path}/orbits -a {proc_path}/aux -d {proc_path}/dem/elevation.dem -w {proc_path}/work -C geometry --bbox '{args['bbox']['S']} {args['bbox']['N']} {args['bbox']['E']} {args['bbox']['W']}' -W slc -C geometry --num_proc {args['cslc_proc']}"], shell=True, capture_output=True, text=True)

# run to CSLCs
subprocess.run([f'{proc_path}/work/run_files/run_01_unpack_topo_reference'], shell=True)
subprocess.run([f'{proc_path}/work/run_files/run_02_unpack_secondary_slc'], shell=True)
subprocess.run([f'{proc_path}/work/run_files/run_03_average_baseline'], shell=True)
subprocess.run([f'{proc_path}/work/run_files/run_04_fullBurst_geo2rdr'], shell=True)
subprocess.run([f'{proc_path}/work/run_files/run_05_fullBurst_resample'], shell=True)
subprocess.run([f'{proc_path}/work/run_files/run_06_extract_stack_valid_region'], shell=True)
subprocess.run([f'{proc_path}/work/run_files/run_07_merge_reference_secondary_slc'], shell=True)
subprocess.run([f'{proc_path}/work/run_files/run_08_grid_baseline'], shell=True)

# remove run files
subprocess.run([f'rm -r {proc_path}/work/run_files'], shell=True)
# rename to avoid aborting second topsStack run
subprocess.run([f'mv {proc_path}/work/coreg_secondarys {proc_path}/work/coreg_secondarys_tmp'], shell=True)
subprocess.run([f'mv {proc_path}/work/merged {proc_path}/work/merged_cslcs'], shell=True)

# stacksentinel to generate run files for interferogram workflow
# NOTE: do not change work directory name to anything other than "work" or things will break
subprocess.run([f"stackSentinel.py -s {proc_path}/slc -o {proc_path}/work/orbits -a {proc_path}/aux -d {proc_path}/dem/elevation.dem -w {proc_path}/work -C geometry --bbox '{args['bbox']['S']} {args['bbox']['N']} {args['bbox']['E']} {args['bbox']['W']}' -W interferogram -C geometry --num_proc {args['insar_proc']} -c {args['insar_connections']} -z {args['azimuth_looks']} -r {args['range_looks']}"], shell=True)

# replace coreg_secondarys 
subprocess.run([f'mv {proc_path}/work/coreg_secondarys_tmp {proc_path}/work/coreg_secondarys'], shell=True)

# run to wrapped interferograms
subprocess.run([f'{proc_path}/work/run_files/run_07_merge_reference_secondary_slc'], shell=True)
subprocess.run([f'{proc_path}/work/run_files/run_08_generate_burst_igram'], shell=True)
subprocess.run([f'{proc_path}/work/run_files/run_09_merge_burst_igram'], shell=True)
subprocess.run([f'{proc_path}/work/run_files/run_10_filter_coherence'], shell=True)

# --------------------------------------------------------------------------
# autoRIFT 
# Import required packages
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from datetime import datetime, timedelta
import time
import rasterio as rio
import xarray as xr
import rioxarray
import isce
import logging
root_logger = logging.getLogger()
root_logger.setLevel('WARNING')
import logging
from imageMath import IML
import isce
from components.contrib.geo_autoRIFT.autoRIFT import autoRIFT_ISCE
from components.contrib.geo_autoRIFT.autoRIFT import __version__ as version
import isceobj
import subprocess
from scipy.interpolate import interpn

# set environment variables
os.environ['AUTORIFT'] = '/mnt/Backups/gbrench/sw/insar_tools/isce2/src/isce2/contrib/geo_autoRIFT'
os.environ['PYTHONPATH'] = os.environ['AUTORIFT']
os.environ['PATH'] += f":{os.environ['AUTORIFT']}"

def select_pairs(scene_list, min_temp_bline, max_temp_bline):
    """
    Create pairs from a list of S1 granules, given a minimum and maximum temporal baseline.
    Returns a list containing lists of granules that fit the criteria. 
    """
    scene_dates = {}
    for scene in scene_list:
        date = scene[17:25]
        scene_dates[date] = scene

    pairs = []
    for date1 in scene_dates:
        for date2 in scene_dates:
            if datetime.strptime(date2, '%Y%m%d')-datetime.strptime(date1, '%Y%m%d') < timedelta(days=max_temp_bline) and not datetime.strptime(date2, '%Y%m%d')-datetime.strptime(date1, '%Y%m%d') < timedelta(days=min_temp_bline) and not date1 >= date2:
                pairs.append([date1, date2])
    
    print(f'number of pairs: {len(pairs)}')
    
    return pairs 

# will want to expose these choices
pairs = select_pairs(args['scene_list'], args['min_temp_baseline'], args['max_temp_baseline'])

def run_autoRIFT(pair, skip_x=3, skip_y=18, scale_y_chip=6, min_azm_chip=16, max_azm_chip=64,
                 preproc_filter_width=21, mpflag=10, search_limit_x=4, search_limit_y=20):

    """
    Runs autoRIFT on two coregistered S1 SLC products. 
    """

    print(f'opening pair: {pair[0]}-{pair[1]}')

    reference_fn = f'{proc_path}/work/merged_cslcs/SLC/{pair[0]}/{pair[0]}.slc.full'
    secondary_fn = f'{proc_path}/work/merged_cslcs/SLC/{pair[1]}/{pair[1]}.slc.full'

    ds1 = gdal.Open(reference_fn, gdal.GA_ReadOnly)
    slc1 = ds1.GetRasterBand(1).ReadAsArray()

    ds2 = gdal.Open(secondary_fn, gdal.GA_ReadOnly)
    slc2 = ds2.GetRasterBand(1).ReadAsArray()

    I1 = np.abs(slc1)
    I2 = np.abs(slc2)
        
    obj = autoRIFT_ISCE()
    obj.configure()
    obj.MultiThread = mpflag

    # rotate to take advantage of chip size scaling
    I1 = np.rot90(I1)
    I2 = np.rot90(I2)
    # scale range chip size to get nearly square chip in cartesian coordinates
    obj.ScaleChipSizeY = scale_y_chip

    obj.I1 = I1
    obj.I2 = I2

    obj.SkipSampleX = skip_x
    obj.SkipSampleY = skip_y

    # Kernel sizes to use for correlation
    obj.ChipSizeMinX = min_azm_chip
    obj.ChipSizeMaxX = max_azm_chip
    obj.ChipSize0X = min_azm_chip
    # oversample ratio, balancing precision and performance for different chip sizes
    obj.OverSampleRatio = {obj.ChipSize0X:32,obj.ChipSize0X*2:64,obj.ChipSize0X*4:128}

    # generate grid
    m,n = obj.I1.shape
    xGrid = np.arange(obj.SkipSampleX+10,n-obj.SkipSampleX,obj.SkipSampleX)
    yGrid = np.arange(obj.SkipSampleY+10,m-obj.SkipSampleY,obj.SkipSampleY)
    nd = xGrid.__len__()
    md = yGrid.__len__()
    obj.xGrid = np.int32(np.dot(np.ones((md,1)),np.reshape(xGrid,(1,xGrid.__len__()))))
    obj.yGrid = np.int32(np.dot(np.reshape(yGrid,(yGrid.__len__(),1)),np.ones((1,nd))))
    noDataMask = np.invert(np.logical_and(obj.I1[:, xGrid-1][yGrid-1, ] > 0, obj.I2[:, xGrid-1][yGrid-1, ] > 0))

    # set search limits
    obj.SearchLimitX = np.full_like(obj.xGrid, search_limit_x)
    obj.SearchLimitY = np.full_like(obj.xGrid, search_limit_y)

    # set search limit and offsets in nodata areas
    obj.SearchLimitX = obj.SearchLimitX * np.logical_not(noDataMask)
    obj.SearchLimitY = obj.SearchLimitY * np.logical_not(noDataMask)
    obj.Dx0 = obj.Dx0 * np.logical_not(noDataMask)
    obj.Dy0 = obj.Dy0 * np.logical_not(noDataMask)
    obj.Dx0[noDataMask] = 0
    obj.Dy0[noDataMask] = 0
    obj.NoDataMask = noDataMask

    print("preprocessing")
    obj.WallisFilterWidth = preproc_filter_width
    obj.preprocess_filt_hps()
    obj.uniform_data_type()

    print("starting autoRIFT")
    obj.runAutorift()
    print("autoRIFT complete")

    # return outputs to original orientation
    tmpDx = np.rot90(obj.Dx, axes=(1, 0))
    obj.Dx = np.rot90(obj.Dy, axes=(1, 0))
    obj.Dy = tmpDx
    tmpxGrid = np.rot90(obj.xGrid, axes=(1, 0))
    obj.xGrid = np.rot90(obj.yGrid, axes=(1, 0))
    obj.yGrid = tmpxGrid
    obj.InterpMask = np.rot90(obj.InterpMask, axes=(1, 0))
    obj.ChipSizeX = np.rot90(obj.ChipSizeX, axes=(1, 0))
    obj.NoDataMask = np.rot90(obj.NoDataMask, axes=(1, 0))

    # convert displacement to m
    obj.Dx_m = obj.Dx * 2.3
    obj.Dy_m = obj.Dy * 12.1
        
    return obj

os.makedirs(f'{proc_path}/offsets', exist_ok=True)

# iterate over pairs and run autorift
for i, pair in enumerate(pairs):
    print(f'working on {i+1}/{len(pairs)}')
    if not os.path.exists(f'{proc_path}/offsets/{pair[0]}-{pair[1]}.nc'):
        obj = run_autoRIFT(pair=pair,
                           skip_x=args['azimuth_skips'],
                           skip_y=args['range_skips'],
                           min_azm_chip=args['min_azimuth_chip'],
                           max_azm_chip=args['max_azimuth_chip'],
                           preproc_filter_width=args['preproc_filter_width'],
                           mpflag=args['feature_tracking_proc'],
                           search_limit_x=args['azimuth_search_limit'],
                           search_limit_y=args['range_search_limit'])

        # interpolate to original cslc dimensions 
        x_coords = np.flip(obj.xGrid[0, :])
        y_coords = obj.yGrid[:, 0]
        
        # Create a mesh grid for the slc dimensions
        x_coords_new, y_coords_new = np.meshgrid(
            np.arange(obj.I2.shape[0]),
            np.arange(obj.I2.shape[1])
        )

        # Perform bilinear interpolation using scipy.interpolate.interpn
        Dx_full = interpn((y_coords, x_coords), obj.Dx_m, (y_coords_new, x_coords_new), method="linear", bounds_error=False)
            
        # load range offsets to xarray 
        ds = xr.Dataset({'Dx_m':(['y', 'x'], Dx_full)},
                        {'range':x_coords_new[0, :],
                         'azimuth':y_coords_new[:, 0]})

        # multilook to match igram
        da_multilooked = ds.Dx_m.coarsen(x=args['range_looks'], y=args['azimuth_looks'], boundary='trim').mean()
        
        # save tif for mintpy
        da_multilooked.rio.to_raster(f'{proc_path}/offsets/dx_m_{pair[0]}-{pair[1]}.tif')
    else:
        print('autoRIFT outputs exist, skipping')
    print('--------------------------------')

# ------------------------------------------------------------------------
# SNAPHU
# may want to switch to python snaphu
import xarray as xr
import rasterio as rio
import rioxarray
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal, gdal_array
from skimage import data, filters
from scipy.interpolate import interpn
import os
from glob import glob
from datetime import datetime

def prep_inputs(igram_dir):
    '''
    Crop interferogram and coherence around feature of interest
    '''
    igram_fn = f'{igram_dir}/filt_fine.int'
    cor_fn = f'{igram_dir}/filt_fine.cor'

    igram_ds = gdal.Open(igram_fn, gdal.GA_ReadOnly)
    igram = igram_ds.GetRasterBand(1).ReadAsArray()
    cor_ds = gdal.Open(cor_fn, gdal.GA_ReadOnly)
    cor = cor_ds.GetRasterBand(1).ReadAsArray()

    # crop to area around feature of interest
    igram_aoi = igram[3250:4750, 2600:4100] # will need to change 
    cor_aoi = cor[3250:4750, 2600:4100]

    # save cropped interferogram
    igram_aoi = np.nan_to_num(igram_aoi, nan=0)
    igram_aoi = igram_aoi.astype(np.complex64)
    output_file = f'{igram_dir}/filt_fine_aoi.int' 
    driver_format = "ISCE"    # Specify the GDAL format for the output (GeoTIFF in this example)
    rows, cols = igram_aoi.shape    # Get the number of rows and columns from the NumPy array
    data_type = gdal_array.NumericTypeCodeToGDALTypeCode(igram_aoi.dtype)  # Convert NumPy data type to GDAL data type
    driver = gdal.GetDriverByName(driver_format)
    output_ds = driver.Create(output_file, cols, rows, 1, data_type)
    band = output_ds.GetRasterBand(1)  
    band.WriteArray(igram_aoi)
    output_ds = None

    # save cropped coherence
    cor_aoi = np.nan_to_num(cor_aoi, nan=0)
    cor_aoi = cor_aoi.astype(np.float32)
    output_file = f'{igram_dir}/filt_fine_aoi.cor' 
    driver_format = "ISCE"    # Specify the GDAL format for the output (GeoTIFF in this example)
    rows, cols = cor_aoi.shape    # Get the number of rows and columns from the NumPy array
    data_type = gdal_array.NumericTypeCodeToGDALTypeCode(cor_aoi.dtype)  # Convert NumPy data type to GDAL data type
    driver = gdal.GetDriverByName(driver_format)
    output_ds = driver.Create(output_file, cols, rows, 1, data_type)
    band = output_ds.GetRasterBand(1) 
    band.WriteArray(cor_aoi)
    output_ds = None

# Function to write to snaphu config file
def write_config_file(out_file, CONFIG_TXT, mode='a'): 
    """Write text files"""
    if not os.path.isfile(out_file) or mode == 'w':
        with open(out_file, "w") as fid:
            fid.write(CONFIG_TXT)
        print('write to file: {}'.format(out_file))
    else:
        with open(out_file, "a") as fid:
            fid.write("\n" + CONFIG_TXT)
        print('add the following to file: \n{}'.format(CONFIG_TXT))

CONFIG_TXT = f'''# snaphu configuration file
#############################################
# File input and output and runtime options #
#############################################

# Input file name
INFILE	filt_fine_aoi.int

# Input file line length 
# will need to grab from file!
LINELENGTH	1500 

# Output file name
OUTFILE	filt_fine_aoi.unw

# Correlation file name
CORRFILE	filt_fine_aoi.cor

# Text file to which runtime parameters will be logged.  
LOGFILE       snaphu.logfile

# Statistical-cost mode (TOPO, DEFO, SMOOTH, or NOSTATCOSTS)
STATCOSTMODE	SMOOTH

# Algorithm used for initialization of wrapped phase values.  Possible
# values are MST and MCF.  
INITMETHOD	MCF

################
# File formats #
################

# Input file format
INFILEFORMAT		COMPLEX_DATA

# Output file format
OUTFILEFORMAT		FLOAT_DATA

# Correlation file format
CORRFILEFORMAT		FLOAT_DATA

###############################
# Connected component control #
###############################

# Grow connected components mask and write to the output file whose
# name is specified here as a string.  The mask is a file of unsigned
# integer values with the same number of rows and columns as the
# unwrapped interferogram.  The type of integer (1 byte vs. 4 byte) is
# specified by the CONNCOMPOUTTYPE keyword, with 1-byte integers being
# the default.
CONNCOMPFILE            conn_comp

# Minimum size of a single connected component, as a fraction (double)
# of the total number of pixels in tile.
MINCONNCOMPFRAC 	0.00001

# End of snaphu configuration file'''
config_file = 'snaphu.conf.brief'

# xml for unw file. Will need to grab line length 
unwrapped_xml_txt = '''
<imageFile>
  <property name="WIDTH">
    <value>1500</value>
  </property>
  <property name="LENGTH">
    <value>1500</value>
  </property>
  <property name="NUMBER_BANDS">
    <value>1</value>
  </property>
  <property name="DATA_TYPE">
    <value>FLOAT</value>
  </property>
  <property name="SCHEME">
    <value>BIP</value>
  </property>
  <property name="BYTE_ORDER">
    <value>l</value>
  </property>
  <property name="ACCESS_MODE">
    <value>read</value>
  </property>
  <property name="FILE_NAME">
    <value>filt_fine_aoi.unw</value>
  </property>
  <component name="Coordinate1">
    <factorymodule>isceobj.Image</factorymodule>
    <factoryname>createCoordinate</factoryname>
    <doc>First coordinate of a 2D image (width).</doc>
    <property name="name">
      <value>ImageCoordinate_name</value>
    </property>
    <property name="family">
      <value>ImageCoordinate</value>
    </property>
    <property name="size">
      <value>1500</value>
    </property>
  </component>
  <component name="Coordinate2">
    <factorymodule>isceobj.Image</factorymodule>
    <factoryname>createCoordinate</factoryname>
    <property name="name">
      <value>ImageCoordinate_name</value>
    </property>
    <property name="family">
      <value>ImageCoordinate</value>
    </property>
    <property name="size">
      <value>1500</value>
    </property>
  </component>
</imageFile>
'''
unwrapped_xml_file = 'filt_fine_aoi.unw.xml'

# xml for connected component file. Will also need to grab line length
conn_comp_xml_txt = '''
<imageFile>
  <property name="WIDTH">
    <value>1500</value>
  </property>
  <property name="LENGTH">
    <value>1500</value>
  </property>
  <property name="NUMBER_BANDS">
    <value>1</value>
  </property>
  <property name="DATA_TYPE">
    <value>BYTE</value>
  </property>
  <property name="SCHEME">
    <value>BIP</value>
  </property>
  <property name="BYTE_ORDER">
    <value>l</value>
  </property>
  <property name="ACCESS_MODE">
    <value>read</value>
  </property>
  <property name="FILE_NAME">
    <value>conn_comp</value>
  </property>
  <component name="Coordinate1">
    <factorymodule>isceobj.Image</factorymodule>
    <factoryname>createCoordinate</factoryname>
    <doc>First coordinate of a 2D image (width).</doc>
    <property name="name">
      <value>ImageCoordinate_name</value>
    </property>
    <property name="family">
      <value>ImageCoordinate</value>
    </property>
    <property name="size">
      <value>1500</value>
    </property>
  </component>
  <component name="Coordinate2">
    <factorymodule>isceobj.Image</factorymodule>
    <factoryname>createCoordinate</factoryname>
    <property name="name">
      <value>ImageCoordinate_name</value>
    </property>
    <property name="family">
      <value>ImageCoordinate</value>
    </property>
    <property name="size">
      <value>1500</value>
    </property>
  </component>
</imageFile>
'''
conn_comp_xml_file = 'conn_comp.xml'

# list interferograms (will need to edit path)
igram_list = glob(f'{proc_path}/work/merged/interferograms/*')

# unwrap interferograms 
for i, igram_dir in enumerate(igram_list):
    date = os.path.basename(igram_dir)
    print(f'working on {date}, {i+1}/{len(igram_list)}')

    tbaseline = datetime.strptime(date[9:17], '%Y%m%d')-datetime.strptime(date[0:8], '%Y%m%d')
    print(f'temporal baseline: {tbaseline.days} days')
    
    os.chdir(igram_dir)
    if not os.path.exists(f'{igram_dir}/filt_fine_aoi.unw.xml'):
        print('prepping inputs')
        prep_inputs(igram_dir)
        write_config_file(config_file, CONFIG_TXT, mode='w')
        print('************unwrapping************')
        subprocess.run(['snaphu -f snaphu.conf.brief'])
        write_config_file(unwrapped_xml_file, unwrapped_xml_txt, mode='w')
        write_config_file(conn_comp_xml_file, conn_comp_xml_txt, mode='w')
        !rm *.rsc
    else:
       print('unwrapped igram exists, skipping')
    print('--------------------------------------------------')

#-----------------------------------------------------------------------
# standardize displacement products for fusion
import xarray as xr
import rasterio as rio
import rioxarray
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal, gdal_array
import os
import shutil
from glob import glob
from scipy.interpolate import interpn

# will need to fix paths
igram_list = glob(f'{proc_path}/work/merged/interferograms/*')
range_offset_list = glob(f'{proc_path}/offsets/*.tif')

os.makedirs(f'{proc_path}/fusion_pairs', exist_ok=True)
fusion_pair_path = f'{proc_path}/fusion_pairs'

# copy all igram pairs to fuse folder
for igram_dir in igram_list:
    shutil.copytree(igram_dir, f'{fusion_pair_path}/{os.path.basename(igram_dir)}_interferogram')

# copy all offset pairs to fuse folder
for range_offset in range_offset_list:
    dates = f'{os.path.basename(range_offset)[5:13]}_{os.path.basename(range_offset)[14:22]}'
    os.makedirs(f'{fusion_pair_path}/{dates}_offset')
    shutil.copy(range_offset, f'{fusion_pair_path}/{dates}_offset/range_offset.tif')

# NOTE -- don't want to run mintpy just to get average coherence. Should calculate from interferograms (if that's what we end up using)
# open insar coherence
t_cor_fn = '/mnt/Backups/gbrench/repos/fusits/nbs/imja/agu_push/AT12/mintpy_igrams/temporalCoherence.h5'
t_cor_ds = xr.open_dataset(t_cor_fn)
t_phony_coherence = 1 - t_cor_ds.temporalCoherence.values

offset_dir_list = glob(f'{fusion_pair_path}/*_offset')
#igram_dir = glob(f'{fusion_pair_path}/*_interferogram')[0]

for offset_dir in offset_dir_list:
    os.chdir(offset_dir)

    offset_fn = f'{offset_dir}/range_offset.tif'
    offset_ds = gdal.Open(offset_fn, gdal.GA_ReadOnly)
    offset = offset_ds.GetRasterBand(1).ReadAsArray()
    offset = np.nan_to_num(offset, nan=0)

    offset_phase = offset*(12.5663706/0.05546576)

    offset_aoi = offset_phase[3250:4750, 2600:4100]

    # write cropped feature tracking displacement
    offset_aoi = np.where(phony_coherence > args['feature_tracking_coherence_mask_threshold'], offset_aoi, 0)
    offset_aoi = offset_aoi.astype(np.float32)
    output_file = f'{offset_dir}/phony_filt_fine_aoi_noest.unw' 
    driver_format = "ISCE"    # Specify the GDAL format for the output (GeoTIFF in this example)
    rows, cols = offset_aoi.shape    # Get the number of rows and columns from the NumPy array
    data_type = gdal_array.NumericTypeCodeToGDALTypeCode(offset_aoi.dtype)  # Convert NumPy data type to GDAL data type
    driver = gdal.GetDriverByName(driver_format)
    output_ds = driver.Create(output_file, cols, rows, 1, data_type)
    band = output_ds.GetRasterBand(1)  
    band.WriteArray(offset_aoi)
    output_ds = None

    # write phony coherence 
    #coherence = np.ones_like(offset_aoi)
    output_file = f'{offset_dir}/phony_filt_fine_aoi.cor' 
    driver_format = "ISCE"    # Specify the GDAL format for the output (GeoTIFF in this example)
    rows, cols = phony_coherence.shape    # Get the number of rows and columns from the NumPy array
    data_type = gdal_array.NumericTypeCodeToGDALTypeCode(phony_coherence.dtype)  # Convert NumPy data type to GDAL data type
    driver = gdal.GetDriverByName(driver_format)
    output_ds = driver.Create(output_file, cols, rows, 1, data_type)
    band = output_ds.GetRasterBand(1)  
    band.WriteArray(phony_coherence)
    output_ds = None

#--------------------------------------------------------------------------------
# MintPy
# Import required packages
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal, gdal_array
from datetime import datetime, timedelta
import xarray as xr
import rasterio as rio
import rioxarray
import geopandas as gpd
import time
from glob import glob
import scipy.signal
from mintpy.utils import readfile, writefile, utils as ut, plot
from mintpy.cli import view, tsview, plot_network, plot_transection
from mintpy.view import prep_slice, plot_slice
from pathlib import Path

os.makedirs(f'{proc_path}/mintpy_fusion', exist_ok=True)
mintpy_path = './data/mintpy_fusion'

CONFIG_TXT = f'''
# vim: set filetype=cfg:
##------------------------ smallbaselineApp.cfg ------------------------##
########## computing resource configuration
mintpy.compute.maxMemory = {args['max_memory']} #auto for 4, max memory to allocate in GB
## parallel processing with dask
## currently apply to steps: invert_network, correct_topography
## cluster   = none to turn off the parallel computing
## numWorker = all  to use all of locally available cores (for cluster = local only)
## numWorker = 80%  to use 80% of locally available cores (for cluster = local only)
## config    = none to rollback to the default name (same as the cluster type; for cluster != local)
mintpy.compute.cluster   = local #[local / slurm / pbs / lsf / none], auto for none, cluster type
mintpy.compute.numWorker = {args['num_workers']} #[int > 1 / all / num%], auto for 4 (local) or 40 (slurm / pbs / lsf), num of workers
mintpy.compute.config    = auto #[none / slurm / pbs / lsf ], auto for none (same as cluster), config name

########## 1. load_data
##---------add attributes manually
## MintPy requires attributes listed at: https://mintpy.readthedocs.io/en/latest/api/attributes/
## Missing attributes can be added below manually (uncomment #), e.g.
# ORBIT_DIRECTION = ascending
# PLATFORM = Sen
# ...
## a. autoPath - automatic path pattern defined in mintpy.defaults.auto_path.AUTO_PATH_*
## b. load_data.py -H to check more details and example inputs.
## c. compression to save disk usage for ifgramStack.h5 file:
## no   - save   0% disk usage, fast [default]
## lzf  - save ~57% disk usage, relative slow
## gzip - save ~62% disk usage, very slow [not recommend]
mintpy.load.processor       = isce  #[isce, aria, hyp3, gmtsar, snap, gamma, roipac, nisar], auto for isce
mintpy.load.autoPath        = auto  #[yes / no], auto for no, use pre-defined auto path
mintpy.load.updateMode      = auto  #[yes / no], auto for yes, skip re-loading if HDF5 files are complete
mintpy.load.compression     = auto  #[gzip / lzf / no], auto for no.
##---------for ISCE only:
mintpy.load.metaFile        = {proc_path}/work/reference/IW1.xml  #[path of common metadata file for the stack], i.e.: ./reference/IW1.xml, ./referenceShelve/data.dat
mintpy.load.baselineDir     = {proc_path}/work/baselines  #[path of the baseline dir], i.e.: ./baselines
##---------interferogram stack:
mintpy.load.unwFile         = {proc_path}/fusion_pairs/*/*filt_fine_aoi_noest.unw  #[path pattern of unwrapped interferogram files]
mintpy.load.corFile         = {proc_path}/fusion_pairs/*/*filt_fine_aoi.cor  #[path pattern of spatial coherence       files]
mintpy.load.connCompFile    = {proc_path}/fusion_pairs/*/*conn_comp  #[path pattern of connected components    files], optional but recommended
mintpy.load.intFile         = auto  #[path pattern of wrapped interferogram   files], optional
mintpy.load.magFile         = auto  #[path pattern of interferogram magnitude files], optional
##---------geometry:
mintpy.load.demFile         = {proc_path}/work/merged/geom_reference/hgt_aoi.rdr  #[path of DEM file]
mintpy.load.lookupYFile     = {proc_path}/work/merged/geom_reference/lat_aoi.rdr  #[path of latitude /row   /y coordinate file], not required for geocoded data
mintpy.load.lookupXFile     = {proc_path}/work/merged/geom_reference/lon_aoi.rdr  #[path of longitude/column/x coordinate file], not required for geocoded data
mintpy.load.incAngleFile    = {proc_path}/work/merged/geom_reference/incLocal_aoi.rdr  #[path of incidence angle file], optional but recommended
mintpy.load.azAngleFile     = {proc_path}/work/merged/geom_reference/los_aoi.rdr  #[path of azimuth   angle file], optional
mintpy.load.shadowMaskFile  = {proc_path}/work/merged/geom_reference/shadowMask_aoi.rdr  #[path of shadow mask file], optional but recommended

########## 2. modify_network
## 1) Network modification based on temporal/perpendicular baselines, date, num of connections etc.
mintpy.network.tempBaseMax     = auto  #[1-inf, no], auto for no, max temporal baseline in days
mintpy.network.perpBaseMax     = {args['max_perp_baseline']}  #[1-inf, no], auto for no, max perpendicular spatial baseline in meter
mintpy.network.connNumMax      = auto  #[1-inf, no], auto for no, max number of neighbors for each acquisition
mintpy.network.startDate       = auto  #[20090101 / no], auto for no
mintpy.network.endDate         = auto  #[20110101 / no], auto for no
mintpy.network.excludeDate     = auto  #[20080520,20090817 / no], auto for no
mintpy.network.excludeIfgIndex = auto  #[1:5,25 / no], auto for no, list of ifg index (start from 0)
mintpy.network.referenceFile   = auto  #[date12_list.txt / ifgramStack.h5 / no], auto for no

## 2) Data-driven network modification
## a - Coherence-based network modification = (threshold + MST) by default
## reference: Yunjun et al. (2019, section 4.2 and 5.3.1); Chaussard et al. (2015, GRL)
## It calculates a average coherence for each interferogram using spatial coherence based on input mask (with AOI)
## Then it finds a minimum spanning tree (MST) network with inverse of average coherence as weight (keepMinSpanTree)
## Next it excludes interferograms if a) the average coherence < minCoherence AND b) not in the MST network.
mintpy.network.coherenceBased  = auto  #[yes / no], auto for no, exclude interferograms with coherence < minCoherence
mintpy.network.minCoherence    = auto  #[0.0-1.0], auto for 0.7

## b - Effective Coherence Ratio network modification = (threshold + MST) by default
## reference: Kang et al. (2021, RSE)
## It calculates the area ratio of each interferogram that is above a spatial coherence threshold.
## This threshold is defined as the spatial coherence of the interferograms within the input mask.
## It then finds a minimum spanning tree (MST) network with inverse of the area ratio as weight (keepMinSpanTree)
## Next it excludes interferograms if a) the area ratio < minAreaRatio AND b) not in the MST network.
mintpy.network.areaRatioBased  = auto  #[yes / no], auto for no, exclude interferograms with area ratio < minAreaRatio
mintpy.network.minAreaRatio    = auto  #[0.0-1.0], auto for 0.75

## Additional common parameters for the 2) data-driven network modification
mintpy.network.keepMinSpanTree = auto  #[yes / no], auto for yes, keep interferograms in Min Span Tree network
mintpy.network.maskFile        = auto  #[file name, no], auto for waterMask.h5 or no [if no waterMask.h5 found]
mintpy.network.aoiYX           = auto  #[y0:y1,x0:x1 / no], auto for no, area of interest for coherence calculation
mintpy.network.aoiLALO         = auto  #[S:N,W:E / no], auto for no - use the whole area

########## 3. reference_point
## Reference all interferograms to one common point in space
## auto - randomly select a pixel with coherence > minCoherence
## however, manually specify using prior knowledge of the study area is highly recommended
##   with the following guideline (section 4.3 in Yunjun et al., 2019):
## 1) located in a coherence area, to minimize the decorrelation effect.
## 2) not affected by strong atmospheric turbulence, i.e. ionospheric streaks
## 3) close to and with similar elevation as the AOI, to minimize the impact of spatially correlated atmospheric delay
mintpy.reference.yx            = auto   #[257,151 / auto]
mintpy.reference.lalo          = {args['stable_reference']}   #[31.8,130.8 / auto]
mintpy.reference.maskFile      = auto   #[filename / no], auto for maskConnComp.h5
mintpy.reference.coherenceFile = auto   #[filename], auto for avgSpatialCoh.h5
mintpy.reference.minCoherence  = auto   #[0.0-1.0], auto for 0.85, minimum coherence for auto method

########## quick_overview
## A quick assessment of:
## 1) possible groud deformation
##    using the velocity from the traditional interferogram stacking
##    reference: Zebker et al. (1997, JGR)
## 2) distribution of phase unwrapping error
##    from the number of interferogram triplets with non-zero integer ambiguity of closue phase
##    reference: T_int in Yunjun et al. (2019, CAGEO). Related to section 3.2, equation (8-9) and Fig. 3d-e.


########## 4. correct_unwrap_error (optional)
## connected components (mintpy.load.connCompFile) are required for this step.
## SNAPHU (Chem & Zebker,2001) is currently the only unwrapper that provides connected components as far as we know.
## reference: Yunjun et al. (2019, section 3)
## supported methods:
## a. phase_closure          - suitable for highly redundant network
## b. bridging               - suitable for regions separated by narrow decorrelated features, e.g. rivers, narrow water bodies
## c. bridging+phase_closure - recommended when there is a small percentage of errors left after bridging
mintpy.unwrapError.method          = no  #[bridging / phase_closure / bridging+phase_closure / no], auto for no
mintpy.unwrapError.waterMaskFile   = auto  #[waterMask.h5 / no], auto for waterMask.h5 or no [if not found]
mintpy.unwrapError.connCompMinArea = 500  #[1-inf], auto for 2.5e3, discard regions smaller than the min size in pixels

## phase_closure options:
## numSample - a region-based strategy is implemented to speedup L1-norm regularized least squares inversion.
##     Instead of inverting every pixel for the integer ambiguity, a common connected component mask is generated,
##     for each common conn. comp., numSample pixels are radomly selected for inversion, and the median value of the results
##     are used for all pixels within this common conn. comp.
mintpy.unwrapError.numSample       = 100  #[int>1], auto for 100, number of samples to invert for common conn. comp.

## bridging options:
## ramp - a phase ramp could be estimated based on the largest reliable region, removed from the entire interferogram
##     before estimating the phase difference between reliable regions and added back after the correction.
## bridgePtsRadius - half size of the window used to calculate the median value of phase difference
mintpy.unwrapError.ramp            = auto  #[linear / quadratic], auto for no; recommend linear for L-band data
mintpy.unwrapError.bridgePtsRadius = auto  #[1-inf], auto for 50, half size of the window around end points

########## 5. invert_network
## Invert network of interferograms into time-series using weighted least square (WLS) estimator.
## weighting options for least square inversion [fast option available but not best]:
## a. var - use inverse of covariance as weight (Tough et al., 1995; Guarnieri & Tebaldini, 2008) [recommended]
## b. fim - use Fisher Information Matrix as weight (Seymour & Cumming, 1994; Samiei-Esfahany et al., 2016).
## c. coh - use coherence as weight (Perissin & Wang, 2012)
## d. no  - uniform weight (Berardino et al., 2002) [fast]
## SBAS (Berardino et al., 2002) = minNormVelocity (yes) + weightFunc (no)
mintpy.networkInversion.weightFunc      = coh #[var / fim / coh / no], auto for var
mintpy.networkInversion.waterMaskFile   = auto #[filename / no], auto for waterMask.h5 or no [if not found]
mintpy.networkInversion.minNormVelocity = auto #[yes / no], auto for yes, min-norm deformation velocity / phase

## mask options for unwrapPhase of each interferogram before inversion (recommend if weightFunct=no):
## a. coherence              - mask out pixels with spatial coherence < maskThreshold
## b. connectComponent       - mask out pixels with False/0 value
## c. no                     - no masking [recommended].
## d. range/azimuthOffsetStd - mask out pixels with offset std. dev. > maskThreshold [for offset]
mintpy.networkInversion.maskDataset   = auto #[coherence / connectComponent / rangeOffsetStd / azimuthOffsetStd / no], auto for no
mintpy.networkInversion.maskThreshold = auto #[0-inf], auto for 0.4
mintpy.networkInversion.minRedundancy = auto #[1-inf], auto for 1.0, min num_ifgram for every SAR acquisition

## Temporal coherence is calculated and used to generate the mask as the reliability measure
## reference: Pepe & Lanari (2006, IEEE-TGRS)
mintpy.networkInversion.minTempCoh  = 0.0 #[0.0-1.0], auto for 0.7, min temporal coherence for mask
mintpy.networkInversion.minNumPixel = auto #[int > 1], auto for 100, min number of pixels in mask above
mintpy.networkInversion.shadowMask  = yes #[yes / no], auto for yes [if shadowMask is in geometry file] or no.

########## 6. correct_troposphere (optional but recommended)
## correct tropospheric delay using the following methods:
## a. height_correlation - correct stratified tropospheric delay (Doin et al., 2009, J Applied Geop)
## b. pyaps - use Global Atmospheric Models (GAMs) data (Jolivet et al., 2011; 2014)
##      ERA5  - ERA5    from ECMWF [need to install PyAPS from GitHub; recommended and turn ON by default]
##      MERRA - MERRA-2 from NASA  [need to install PyAPS from Caltech/EarthDef]
##      NARR  - NARR    from NOAA  [need to install PyAPS from Caltech/EarthDef; recommended for N America]
## c. gacos - use GACOS with the iterative tropospheric decomposition model (Yu et al., 2018, JGR)
##      need to manually download GACOS products at http://www.gacos.net for all acquisitions before running this step
mintpy.troposphericDelay.method = no  #[pyaps / height_correlation / gacos / no], auto for pyaps

## Notes for pyaps:
## a. GAM data latency: with the most recent SAR data, there will be GAM data missing, the correction
##    will be applied to dates with GAM data available and skipped for the others.
## b. WEATHER_DIR: if you define an environment variable named WEATHER_DIR to contain the path to a
##    directory, then MintPy applications will download the GAM files into the indicated directory.
##    MintPy application will look for the GAM files in the directory before downloading a new one to
##    prevent downloading multiple copies if you work with different dataset that cover the same date/time.
mintpy.troposphericDelay.weatherModel = auto  #[ERA5 / MERRA / NARR], auto for ERA5
mintpy.troposphericDelay.weatherDir   = auto  #[path2directory], auto for WEATHER_DIR or "./"

## Notes for height_correlation:
## Extra multilooking is applied to estimate the empirical phase/elevation ratio ONLY.
## For an dataset with 5 by 15 looks, looks=8 will generate phase with (5*8) by (15*8) looks
## to estimate the empirical parameter; then apply the correction to original phase (with 5 by 15 looks),
## if the phase/elevation correlation is larger than minCorrelation.
mintpy.troposphericDelay.polyOrder      = auto  #[1 / 2 / 3], auto for 1
mintpy.troposphericDelay.looks          = auto  #[1-inf], auto for 8, extra multilooking num
mintpy.troposphericDelay.minCorrelation = auto  #[0.0-1.0], auto for 0

## Notes for gacos:
## Set the path below to directory that contains the downloaded *.ztd* files
mintpy.troposphericDelay.gacosDir = auto # [path2directory], auto for "./GACOS"

########## 7. deramp (optional)
## Estimate and remove a phase ramp for each acquisition based on the reliable pixels.
## Recommended for localized deformation signals, i.e. volcanic deformation, landslide and land subsidence, etc.
## NOT recommended for long spatial wavelength deformation signals, i.e. co-, post- and inter-seimic deformation.
mintpy.deramp          = linear  #[no / linear / quadratic], auto for no - no ramp will be removed
mintpy.deramp.maskFile = auto  #[filename / no], auto for maskTempCoh.h5, mask file for ramp estimation

########## 8. correct_topography (optional but recommended)
## Topographic residual (DEM error) correction
## reference: Fattahi and Amelung (2013, IEEE-TGRS)
## stepFuncDate      - specify stepFuncDate option if you know there are sudden displacement jump in your area,
##                     e.g. volcanic eruption, or earthquake
## excludeDate       - dates excluded for the error estimation
## pixelwiseGeometry - use pixel-wise geometry (incidence angle & slant range distance)
##                     yes - use pixel-wise geometry if they are available [slow; used by default]
##                     no  - use the mean   geometry [fast]
mintpy.topographicResidual                   = auto  #[yes / no], auto for yes
mintpy.topographicResidual.polyOrder         = auto  #[1-inf], auto for 2, poly order of temporal deformation model
mintpy.topographicResidual.phaseVelocity     = auto  #[yes / no], auto for no - use phase velocity for minimization
mintpy.topographicResidual.stepFuncDate      = auto  #[20080529,20190704T1733 / no], auto for no, date of step jump
mintpy.topographicResidual.excludeDate       = auto  #[20070321 / txtFile / no], auto for exclude_date.txt
mintpy.topographicResidual.pixelwiseGeometry = auto  #[yes / no], auto for yes, use pixel-wise geometry info

########## 9.1 residual_RMS (root mean squares for noise evaluation)
## Calculate the Root Mean Square (RMS) of residual phase time-series for each acquisition
## reference: Yunjun et al. (2019, section 4.9 and 5.4)
## To get rid of long spatial wavelength component, a ramp is removed for each acquisition
## Set optimal reference date to date with min RMS
## Set exclude dates (outliers) to dates with RMS > cutoff * median RMS (Median Absolute Deviation)
mintpy.residualRMS.maskFile = auto  #[file name / no], auto for maskTempCoh.h5, mask for ramp estimation
mintpy.residualRMS.deramp   = auto  #[quadratic / linear / no], auto for quadratic
mintpy.residualRMS.cutoff   = auto  #[0.0-inf], auto for 3

########## 9.2 reference_date
## Reference all time-series to one date in time
## reference: Yunjun et al. (2019, section 4.9)
## no     - do not change the default reference date (1st date)
mintpy.reference.date = no   #[reference_date.txt / 20090214 / no], auto for reference_date.txt

########## 10. velocity
## Estimate a suite of time functions [linear velocity by default]
## from final displacement file (and from tropospheric delay file if exists)
mintpy.timeFunc.startDate   = auto   #[20070101 / no], auto for no
mintpy.timeFunc.endDate     = auto   #[20101230 / no], auto for no
mintpy.timeFunc.excludeDate = auto   #[exclude_date.txt / 20080520,20090817 / no], auto for exclude_date.txt

## Fit a suite of time functions
## reference: Hetland et al. (2012, JGR) equation (2-9)
## polynomial function    is  defined by its degree in integer. 1 for linear, 2 for quadratic, etc.
## periodic   function(s) are defined by a list of periods in decimal years. 1 for annual, 0.5 for semi-annual, etc.
## step       function(s) are defined by a list of onset times in str in YYYYMMDD(THHMM) format
## exp & log  function(s) are defined by an onset time followed by an charateristic time in integer days.
##   Multiple exp and log functions can be overlaied on top of each other, achieved via e.g.:
##   20110311,60,120          - two functions sharing the same onset time OR
##   20110311,60;20170908,120 - separated by ";"
mintpy.timeFunc.polynomial = auto   #[int >= 0], auto for 1, degree of the polynomial function
mintpy.timeFunc.periodic   = auto   #[1,0.5 / list_of_float / no], auto for no, periods in decimal years
mintpy.timeFunc.stepDate   = auto   #[20110311,20170908 / 20120928T1733 / no], auto for no, step function(s)
mintpy.timeFunc.exp        = auto   #[20110311,60 / 20110311,60,120 / 20110311,60;20170908,120 / no], auto for no
mintpy.timeFunc.log        = auto   #[20110311,60 / 20110311,60,120 / 20110311,60;20170908,120 / no], auto for no

## Uncertainty quantification methods:
## a. residue    - propagate from fitting residue assuming normal dist. in time (Fattahi & Amelung, 2015, JGR)
## b. covariance - propagate from time series (co)variance matrix
## c. bootstrap  - bootstrapping (independently resampling with replacement; Efron & Tibshirani, 1986, Stat. Sci.)
mintpy.timeFunc.uncertaintyQuantification = auto   #[residue, covariance, bootstrap], auto for residue
mintpy.timeFunc.timeSeriesCovFile         = auto   #[filename / no], auto for no, time series covariance file
mintpy.timeFunc.bootstrapCount            = auto   #[int>1], auto for 400, number of iterations for bootstrapping

########## 11.1 geocode (post-processing)
# for input dataset in radar coordinates only
# commonly used resolution in meters and in degrees (on equator)
# 100,         90,          60,          50,          40,          30,          20,          10
# 0.000925926, 0.000833334, 0.000555556, 0.000462963, 0.000370370, 0.000277778, 0.000185185, 0.000092593
mintpy.geocode              = auto  #[yes / no], auto for yes
mintpy.geocode.SNWE         = auto  #[-1.2,0.5,-92,-91 / none ], auto for none, output extent in degree
mintpy.geocode.laloStep     = {args['pixel_spacing']}  #[-0.000555556,0.000555556 / None], auto for None, output resolution in degree
mintpy.geocode.interpMethod = auto  #[linear], auto for nearest, interpolation method
mintpy.geocode.fillValue    = auto  #[np.nan, 0, ...], auto for np.nan, fill value for outliers.

########## 11.2 google_earth (post-processing)
mintpy.save.kmz             = auto   #[yes / no], auto for yes, save geocoded velocity to Google Earth KMZ file

########## 11.3 hdfeos5 (post-processing)
mintpy.save.hdfEos5         = auto   #[yes / no], auto for no, save time-series to HDF-EOS5 format
mintpy.save.hdfEos5.update  = auto   #[yes / no], auto for no, put XXXXXXXX as endDate in output filename
mintpy.save.hdfEos5.subset  = auto   #[yes / no], auto for no, put subset range info   in output filename

########## 11.4 plot
# for high-resolution plotting, increase mintpy.plot.maxMemory
# for fast plotting with more parallelization, decrease mintpy.plot.maxMemory
mintpy.plot           = auto  #[yes / no], auto for yes, plot files generated by default processing to pic folder
mintpy.plot.dpi       = auto  #[int], auto for 150, number of dots per inch (DPI)
mintpy.plot.maxMemory = auto  #[float], auto for 4, max memory used by one call of view.py for plotting.
'''

os.chdir(mintpy_path)
config_file = f'{mintpy_path}/Sen.txt'
write_config_file(config_file, CONFIG_TXT, mode='w')

# run mintpy
subprocess.run(['smallbaselineApp.py Sen.txt --dostep load_data'], shell=True)
subprocess.run(['smallbaselineApp.py Sen.txt --dostep modify_network'], shell=True)
subprocess.run(['smallbaselineApp.py Sen.txt --dostep reference_point'], shell=True)
subprocess.run(['smallbaselineApp.py Sen.txt --dostep quick_overview'], shell=True)
subprocess.run(['smallbaselineApp.py Sen.txt --dostep invert_network'], shell=True)
subprocess.run(['smallbaselineApp.py Sen.txt --dostep correct_topography'], shell=True)
subprocess.run(['smallbaselineApp.py Sen.txt --dostep residual_RMS'], shell=True)
subprocess.run(['smallbaselineApp.py Sen.txt --dostep reference_date'], shell=True)
subprocess.run(['smallbaselineApp.py Sen.txt --dostep velocity'], shell=True)
subprocess.run(['smallbaselineApp.py Sen.txt --dostep geocode'], shell=True)

