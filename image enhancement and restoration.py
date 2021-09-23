import cv2
import math
import numpy as np
import sys

def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()

def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix

def simplest_cb(img, percent):
    assert img.shape[2] == 3
    assert percent > 0 and percent < 100
    half_percent = percent / 200.0
    channels = cv2.split(img)

    out_channels = []
    for channel in channels:
        assert len(channel.shape) == 2
        # [EN] find the low and high precentile values (based on the input percentile)
        # [TR] düşük ve yüksek yüzdelik değerleri girdiye göre buluyoruz
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)

        assert len(flat.shape) == 1
        flat = np.sort(flat)
        n_cols = flat.shape[0]

        low_val  = flat[math.floor(n_cols * half_percent)]
        high_val = flat[math.ceil( n_cols * (1.0 - half_percent))]

        # [EN] saturate below the low percentile and above the high percentile
        # [TR] düşük yüzdeliğin altında ve yüksek yüzdeliğin üstünde doygunluk elde ediyoruz
        thresholded = apply_threshold(channel, low_val, high_val)

        # [EN] scale the channel
        # [TR] ölçeklendiriyoruz
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)

if __name__ == '__main__':
    # [EN] The location of the image has to be adjusted (according to your own computer)
    # [TR] Fotoğrafın konumu ayarlanmalı (kendi bilgisayarınıza göre)
    img = cv2.imread(r'C:\Users\Lenovo\Desktop\images\underwater1.jpg') 

    img = cv2.resize(img, (600,600), 2,2,cv2.INTER_AREA)
    out = simplest_cb(img, 1)

    # [EN] First version of the image
    # [TR] Görüntünün ilk hali
    cv2.imshow("before", img) 

    # [EN] Enhanced version of the image
    # [TR] Görüntünün İyileştirilmiş hali
    cv2.imshow("after", out)
    
cv2.waitKey(0)
