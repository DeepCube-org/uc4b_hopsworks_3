
import datetime
import mosaic.s2_download
import rasterio
from mosaic.dwlulc_download import Inference
import matplotlib.pyplot as plt

bbox = (
    46.00, 
    -16.10,
    46.02, 
    -16.15,
)
start = datetime.datetime(2021, 10, 5)
end = datetime.datetime(2021, 10, 7)
n = 1

mosaic.s2_download.mosaic(bbox = bbox, start = start, end = end, n = n, output = './mosaic.tiff', split_shape = (2,2), mask_clouds = False)


with rasterio.open('mosaic.tiff') as file:
    image = file.read()
image = image.transpose((1,2,0))

inference = Inference(all_bands = True)
lulc_prob = inference.predict(image)

lulc_prob = lulc_prob.argmax(-1)

plt.imshow(image[:, :, [3,2,1]].clip(0, 3000)/3000)
plt.show()
plt.imshow(lulc_prob)
plt.show()


import pdb
pdb.set_trace()

