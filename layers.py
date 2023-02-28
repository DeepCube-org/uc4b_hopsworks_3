from mosaic import s1_download
from mosaic import s2_download
from mosaic import esalulc_download
from mosaic import dem_download
import argparse
import fiona
import re
import datetime
import os
import rasterio

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract layers for training/inference')
    parser.add_argument('--shape_file', required=True, type=str)
    parser.add_argument('--save_dir', required=False, type=str, default = '')
    parser.add_argument('--start_date', required=False, type=str, default = None)
    parser.add_argument('--end_date',   required=False, type=str, default = None)
    parser.add_argument('--n_images',   required=False, type=int, default = 5)
    parser.add_argument('--max_retry',   required=False, type=int, default = 10)

    args = parser.parse_args()

    with fiona.open(args.shape_file) as file:
        
        times = []
        point = file[0]
        for name in point['properties'].keys():
            if(re.match('D[0-9]+', name)): #Timeseries start with the letter 'D'
                new_name = name[1:]
                new_name = datetime.datetime.strptime(new_name, '%Y%m%d')
                new_name = new_name.strftime('%Y-%m-%d')
                times.append(new_name)
        bbox = list(file.bounds)
    times = [datetime.datetime.strptime(time, "%Y-%m-%d") for time in times]

    if(args.start_date is None):
        start = min(times)
    else:
        start = datetime.datetime.strptime(args.start_date, "%Y-%m-%d")
    if(args.end_date is None):
        end = max(times)
    else:
        end = datetime.datetime.strptime(args.end_date, "%Y-%m-%d")
    
    
    bbox[0] = bbox[0]-(0.1)#*x_spread
    bbox[1] = bbox[1]-(0.1)#*y_spread
    bbox[2] = bbox[2]+(0.1)#*x_spread
    bbox[3] = bbox[3]+(0.1)#*y_spread

    print(start, end)
    exit(0)
    esalulc_download.mosaic(bbox = bbox, start = start, end = datetime.datetime.now(), output = os.path.join(args.save_dir,'lulc.tiff'), max_retry = args.max_retry)
    dem_download.mosaic(bbox = bbox, start = start, end = datetime.datetime.now(), output = os.path.join(args.save_dir,'dem.tiff'), max_retry = args.max_retry)
    
    s1_download.mosaic(bbox = bbox, start = start, end = end, output = os.path.join(args.save_dir,'sar.tiff'), n = args.n_images, max_retry=args.max_retry)
    s2_download.mosaic(bbox = bbox, start = start, end = end, output = os.path.join(args.save_dir,'optic.tiff'), n = args.n_images, max_retry = args.max_retry)
    
    with rasterio.open(os.path.join(args.save_dir,'lulc.tiff'), 'r') as file:
        lulc_shape = file.read().shape
    with rasterio.open(os.path.join(args.save_dir,'dem.tiff'), 'r') as file:
        dem_shape = file.read().shape
    with rasterio.open(os.path.join(args.save_dir,'sar.tiff'), 'r') as file:
        sar_shape = file.read().shape
    with rasterio.open(os.path.join(args.save_dir,'optic.tiff'), 'r') as file:
        optic_shape = file.read().shape
    
    print(lulc_shape)
    print(dem_shape)
    print(sar_shape)
    print(optic_shape)
    
    