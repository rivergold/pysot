import sys
sys.path.append('.')
import cv2
import numpy as np
from pysot.datasets.dataset import TrkDataset

if __name__ == '__main__':
    dataset = TrkDataset()
    sample = dataset.__getitem__(1)
    print(sample['template'].shape)
    z_img = np.rollaxis(sample['template'], 0, 3)
    x_img = np.rollaxis(sample['search'], 0, 3)
    cv2.imwrite('./z.jpg', z_img)
    cv2.imwrite('./x.jpg', x_img)
    print(f'z_img shape: {z_img.shape}')
    print(f'x_img shape: {x_img.shape}')
