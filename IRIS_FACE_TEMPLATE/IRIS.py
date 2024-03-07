import warnings
warnings.filterwarnings("ignore")
#%config Completer.use_jedi = False

import cv2
import numpy as np
import glob
import math
import scipy
from time import time
from scipy.spatial import distance
from scipy import signal
from scipy.stats import binom, norm
import matplotlib.pyplot as plt
import pandas as pd

import plotly.express as px

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly import tools
import plotly.graph_objs as go
import plotly.io as pio
init_notebook_mode(connected=True)
#Bloom filters 
#Feature Extraction unprotected two-dimensional binary feature vector
def draw_circle(img, c):
    return cv2.circle(cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB), (c[1], c[0]), round(c[2]), (255,255,255), 1)

# iris circular to rect
def norm_iris(img, c, r_iris):
    [cy, cx, r_pupil, _] = c
    
    # fix r_iris 100
    r_iris = 100
    
    _padding = 0
    img_rect = cv2.linearPolar(img, (cx, cy), r_iris + _padding, cv2.WARP_FILL_OUTLIERS)
    img_rect = cv2.rotate(img_rect, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_rect = cv2.resize(img_rect, (600, r_iris + _padding), interpolation=cv2.INTER_AREA)
    img_rect = img_rect[:(r_iris - r_pupil + _padding * 2), :]
    img_rect = cv2.resize(img_rect, (600, 60), interpolation=cv2.INTER_AREA)

    img_rect = cv2.equalizeHist(img_rect)
    
    return img_rect


# Structure-preserving feature re-arrangement
def encode_iris(arr_polar, arr_noise, minw_length, mult, sigma_f):
    """
    Generate iris template and noise mask from the normalised iris region.
    """
    # convolve with gabor filters
    filterb = gaborconvolve_f(arr_polar, minw_length, mult, sigma_f)
    l = arr_polar.shape[1]
    template = np.zeros([arr_polar.shape[0], 2 * l])
    h = np.arange(arr_polar.shape[0])

    # making the iris template
    mask_noise = np.zeros(template.shape)
    filt = filterb[:, :]

    # quantization and check to see if the phase data is useful
    H1 = np.real(filt) > 0
    H2 = np.imag(filt) > 0

    H3 = np.abs(filt) < 0.0001
    for i in range(l):
        ja = 2 * i

        # biometric template
        template[:, ja] = H1[:, i]
        template[:, ja + 1] = H2[:, i]

    return template, mask_noise

#Bloom filter computation
def gaborconvolve_f(img, minw_length, mult, sigma_f):
    """
    Convolve each row of an imgage with 1D log-Gabor filters.
    """
    rows, ndata = img.shape
    logGabor_f = np.zeros(ndata)
    filterb = np.zeros([rows, ndata], dtype=complex)

    radius = np.arange(ndata/2 + 1) / (ndata/2) / 2
    radius[0] = 1

    # filter wavelength
    wavelength = minw_length

    # radial filter component 
    fo = 1 / wavelength
    logGabor_f[0: int(ndata/2) + 1] = np.exp((-(np.log(radius/fo))**2) /
                                    (2 * np.log(sigma_f)**2))
    logGabor_f[0] = 0

    # convolution for each row
    for r in range(rows):
        signal = img[r, 0:ndata]
        imagefft = np.fft.fft(signal)
        filterb[r, :] = np.fft.ifft(imagefft * logGabor_f)
    
    return filterb

#Bloom Filter based templete protection carried out in three steps
def get_gabor_encoded_img(img, c):
    minw_length = 18
    mult = 1
    sigma_f = 0.5

    img_rect = norm_iris(img, c, c[2] + iris_depth)
    img_gabor_rect, mask_noise = encode_iris(img_rect, np.zeros(img_rect.shape), minw_length, mult, sigma_f)
    img_gabor_rect = (img_gabor_rect * 255).astype(np.uint8)
    
    return img_gabor_rect[:, ::2], img_gabor_rect[:, 1::2]


# get hamming distance of 2 iris rects
def get_hd_of_img_i_j(img0, img1):
    key_points_0 = key_points.copy()
    key_points_1 = []
    hd_map = []
    
    for kp in key_points_0:
        feature_0 = img0[kp[0]-hsp:kp[0]+hsp, kp[1]-hsp:kp[1]+hsp]
        feature_1 = img1[kp[0]-sp:kp[0]+sp, kp[1]-sp:kp[1]+sp]

        # quality check
#         feature_0_mean, feature_1_mean = feature_0.mean(), feature_1.mean() 
#         if((feature_0_mean > 255*0.4 and feature_0_mean < 255*0.6) and (feature_1_mean > 255*0.41 and feature_1_mean < 255*0.59)):
        feature_match_map = cv2.filter2D(feature_1, -1, feature_0/feature_0.sum())
        feature_match_map_0 = cv2.filter2D((255 - feature_1), -1, (255 - feature_0)/(255 - feature_0).sum())
        feature_match_map = (feature_match_map.astype(np.int) + feature_match_map_0.astype(np.int)) / 2

        salt = np.random.rand(feature_match_map.shape[0], feature_match_map.shape[1])
        feature_match_map = feature_match_map.astype(np.float32) + salt

        kp_match_ind = np.unravel_index(feature_match_map.argmax(), feature_match_map.shape)
        kp_match_offset = kp_match_ind - np.array([sp, sp])
        kp_match = np.array(kp) + kp_match_offset

        key_points_1.append(kp_match)

        hd_map.append(feature_match_map[kp_match_ind[0]][kp_match_ind[1]])
        
    select_hd = pd.DataFrame(hd_map)[0] > thresh_hd
    
    return key_points_1, np.sum(select_hd), hd_map


N_tested = 756 # number of tested images
iris_depth = 60 # height of normed iris rect
thresh_hd = int(0.7 * 255)
percentile_hd = .75


files = []
for i in range(1, 109):
    files.extend(sorted(glob.glob('CASIA1/' + str(i) + '/*.jpg')))

print("N# of files which we are extracting features", len(files))

# for f in files:
#     print(f)


cirpupils = pd.read_csv('casia_v1_circle_pupil.csv')
print("",cirpupils)


img_oris = [cv2.imread(_, 0) for _ in files[:N_tested]]

ti = 333

fig, ax = plt.subplots(figsize=(12, 12))
ax.imshow(draw_circle(img_oris[ti], list(cirpupils.iloc[ti])), interpolation='nearest')

list(cirpupils.iloc[ti])


t0, t1 = 2, 4 # index of sample images
print(files[t0])
print(files[t1])

fig, ax = plt.subplots(figsize=(16, 8))
ax.imshow(np.hstack((img_oris[t0], img_oris[t1])), interpolation='nearest')


fig, ax = plt.subplots(figsize=(20, 16))
ax.imshow(np.vstack((norm_iris(img_oris[t0], cirpupils.iloc[t0], cirpupils.iloc[t0][2] + iris_depth), 
                     norm_iris(img_oris[t1], cirpupils.iloc[t1], cirpupils.iloc[t1][2] + iris_depth))), interpolation='nearest')



img_gabor_rects = []
for i in range(len(img_oris)):
    template_ori = get_gabor_encoded_img(img_oris[i], cirpupils.iloc[i])
    template_pat = []
    template_pat.append(np.hstack((template_ori[0], template_ori[0][:, :30])))
    template_pat.append(np.hstack((template_ori[1], template_ori[1][:, :30])))
    
    img_gabor_rects.append(template_pat)
    
    
fig, ax = plt.subplots(figsize=(20, 16))
ax.imshow(np.vstack((img_gabor_rects[t0][0], img_gabor_rects[t1][0])), interpolation='nearest')



h_rect, w_rect = img_gabor_rects[0][0].shape

n_h, n_w = 5, 60 # N feature points on H, W
sp = 30 # size of HD area
hsp = int(sp*0.5) # half of sp

step_h = (h_rect - sp) / (n_h - 1)
step_w = (w_rect - sp) / (n_w - 1)

key_points = []
for i in range(n_h):
    _k_h = i * step_h + sp
    for j in range(n_w):
        _k_w = j * step_w + sp
        key_points.append([int(_k_h), int(_k_w)])
        
print(h_rect, w_rect, step_h, step_w)



# Sample of  Hamming Distance map

kp = key_points[20]

feature_0 = img_gabor_rects[t0][0][kp[0]-hsp:kp[0]+hsp, kp[1]-hsp:kp[1]+hsp]
feature_1 = img_gabor_rects[t1][0][kp[0]-sp:kp[0]+sp, kp[1]-sp:kp[1]+sp]

feature_match_map = cv2.filter2D(feature_1, -1, feature_0/feature_0.sum())

kp_match_ind = np.unravel_index(feature_match_map.argmax(), feature_match_map.shape)
kp_match_offset = kp_match_ind - np.array([hsp, hsp])
kp_match = np.array(kp) + kp_match_offset
print(kp_match, feature_match_map[kp_match_ind[0]][kp_match_ind[1]], feature_0.mean(), feature_1.mean())

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(feature_0, interpolation='nearest')


fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(np.hstack((feature_1, feature_match_map)), interpolation='nearest')


key_points_1, n_selected, hd_map = get_hd_of_img_i_j(img_gabor_rects[t0][0], img_gabor_rects[t1][0])
print(n_selected, np.percentile(hd_map, percentile_hd))


