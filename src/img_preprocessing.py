from re import A
import cv2
import numpy as np
import pickle
from tqdm import tqdm


def preprocessing(img,img_size=64,margin=0,threshold_w=0,threshold_h=0):
    img_white = cv2.bitwise_not(img) # Character area white
    y_min = 0
    y_max = 0
    x_min = 0
    x_max = 0
    # Row processing
    for i in range(img_white.shape[0]):
        if np.sum(img_white[i,:]) > 0:
            y_min = i
            break
    for i in reversed(range(img_white.shape[0])):
        if np.sum(img_white[i,:]) > 0:
            y_max = i+1 # Range is 0~n-1, so add +1 for array index adjustment. Example: img[0:N] becomes 0~N-1, so need +1 to avoid bugs
            break
    # Column processing
    for i in range(img_white.shape[1]):
        if np.sum(img_white[:,i]) > 0:
            x_min = i
            break
    for i in reversed(range(img_white.shape[1])):
        if np.sum(img_white[:,i]) > 0:
            x_max = i+1
            break
    img = img_white[y_min:y_max,x_min:x_max]
    h = img.shape[0]
    w = img.shape[1]
    if (h<threshold_h) or (w<threshold_w):
        return 0
    if margin>0:
        img = np.pad(img,[(margin,margin),(margin,margin)],'constant')
    size = max(w,h)
    ratio = img_size/size # How much to scale
    img_resize = cv2.resize(img, (int(w*ratio),int(h*ratio)),interpolation=cv2.INTER_CUBIC)
    # Determine padding width
    if w > h:
        pad = int((img_size - h*ratio)/2)
        # np.pad() second argument [(top, bottom), (left, right)] for row/column padding
        img_resize = np.pad(img_resize,[(pad,pad),(0,0)],'constant')
    elif h > w:
        pad = int((img_size - w*ratio)/2)
        img_resize = np.pad(img_resize,[(0,0),(pad,pad)],'constant')
    # Finally resize cleanly to 100x100
    img_resize = cv2.resize(img_resize,(img_size,img_size),interpolation=cv2.INTER_CUBIC)
    img_resize = cv2.bitwise_not(img_resize)
    img_resize_ = np.dstack((img_resize,img_resize))
    img_resize = np.dstack((img_resize,img_resize_))
    return img_resize

