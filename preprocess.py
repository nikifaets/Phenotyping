import sys
sys.path.append(r'D:\analysis')
import numpy as np
import clarity.IO as io
from clarity.Data.MakeData import makeAllConvData

if __name__ == '__main__':
    XsinkPath = r'D:\analysis\clusterdnn\data\IBA1\X.bc'
    nucleusImgPath = r'D:\analysis\clusterdnn\images\IBA1\Syto16.tif'
    imgPath = r'D:\analysis\clusterdnn\images\IBA1\IBA1.tif'
    centersPath = r'D:\analysis\clusterdnn\data\IBA1\nuclei_centers.npy'
    BOUND_SIZE = 32
    
    img = io.readData(imgPath)
    nucleusImg = io.readData(nucleusImgPath)
    cell_candidates = np.load(centersPath)
    
    makeAllConvData(XSink=XsinkPath, YSink=None, img=img, negatives=cell_candidates, positives=[], 
                            BOUND_SIZE=BOUND_SIZE, nucleusImg=nucleusImg)