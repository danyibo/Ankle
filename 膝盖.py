# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import os, math
import numpy as np
import nibabel as nib
from glob import glob
from skimage import data, io, filters
import matplotlib.pyplot as plt
from scipy.ndimage import generate_binary_structure
from scipy.ndimage.morphology import binary_dilation, binary_closing,binary_opening
from scipy.ndimage import generate_binary_structure
from skimage.filters import threshold_minimum, threshold_otsu, threshold_local
from MeDIT.Visualization import Imshow3DArray

# %%
# 4-neighbor connectivity
# if zero px neighbor exists, return true
def zero_neighbors(image, pos):
    if image[pos[0]+1, pos[1]]==0 or \
       image[pos[0]-1, pos[1]]==0 or \
       image[pos[0], pos[1]+1]==0 or \
       image[pos[0], pos[1]-1]==0:
        return True
    else:
        return False

def end_points(line, pos):
    n = int(line[pos[0]+1, pos[1]]>0) + \
        int(line[pos[0]-1, pos[1]]>0) + \
        int(line[pos[0], pos[1]+1]>0) + \
        int(line[pos[0], pos[1]-1]>0) + \
        int(line[pos[0]+1, pos[1]-1]>0) + \
        int(line[pos[0]-1, pos[1]-1]>0) + \
        int(line[pos[0]+1, pos[1]+1]>0) + \
        int(line[pos[0]-1, pos[1]+1]>0) 

    if n == 1:
        return True
    else:
        return False

def z_range(binary_image, axis=(0,1)):
    "return non-zero z axis slices (min, max+1)"
    z = np.any(binary_image, axis=axis)     # 只要有一个为真，则返回true
    zmin, zmax = np.where(z)[0][[0, -1]]
    return zmin, zmax+1

def z_nonzero_slices(binary_image, axis=(0,1)):
    z = np.any(binary_image, axis=axis)
    slices = np.where(z)[0]
    return slices

def get_boundary(binary_mask, region_label):
    boundaries = []
    slices = np.arange(*z_range(binary_mask))
    for idx in slices:
        bin_slice = binary_mask[..., idx]
        coord = list(zip(*np.where(bin_slice==region_label))) 
        
        boundary = list(filter(lambda x: zero_neighbors(bin_slice, x), coord))
        boundaries.append(boundary)
    return boundaries, slices

# judge whether line segment has intersection with target region
def intersect(p1, p2, region, nb_points=40, dilation=0):
    """"Return if a line segment between points p1 and p2
    has intersection with region"""
    # If we have 8 intermediate points, we have 8+1=9 spaces between p1 and p2
    x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
    y_spacing = (p2[1] - p1[1]) / (nb_points + 1)

    if dilation>0:
        dil_region = binary_dilation(region, generate_binary_structure(2,1), dilation)
    else:
        dil_region = region
    ret = np.any([dil_region[int(p1[0]+i*x_spacing), int(p1[1]+i*y_spacing)]
           for i in range(1, nb_points+1)])
    return ret

def get_line_segment(p1, p2, nb_points=30):
    """"Return a list of nb_points equally spaced points
    between p1 and p2"""
    x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
    y_spacing = (p2[1] - p1[1]) / (nb_points + 1)

    return [(int(p1[0]+i*x_spacing), int(p1[1]+i*y_spacing))
           for i in range(1, nb_points+1)]

def get_delta_xy(pt, reso, poly_fn, delta_dist, k=None):
    '''
    Get x&y px-wise delta of given pt from specified curve fn
    if k is given, directly use k, ignore poly_fn
    '''
    x = pt[0]
    if k is None:
        k = -1 / poly_fn.deriv(1)(x) # norm direction

    delta_x = delta_dist / math.sqrt(1+k*k)
    delta_y = delta_dist / math.sqrt(1+k*k) * k
    delta_x_px = int(delta_x / reso[0])
    delta_y_px = int(delta_y / reso[1])
    return delta_x_px, delta_y_px

def norm_process(img_path, roi_path, out_label):
    image_nii = nib.load(img_path)
    label_nii = nib.load(roi_path)
    image = np.copy(image_nii.get_data())
    label = np.copy(label_nii.get_data())
    reso = image_nii.header['pixdim'][1:4]
    assert image.shape == label.shape, 'Image shape != Label shape'
    print('Image size:', image.shape, 'image reso:', reso)

    # Get threshold for boundary
    thresh_min = threshold_minimum(image[label>0])
    thresh_otsu = threshold_otsu(image[label>0])

    # ax = plt.hist(image[label>0].ravel(), bins = 64)
    # plt.axvline(thresh_min, color='r')
    # plt.axvline(thresh_otsu, color='g')
    # plt.show()

    region_1 = np.logical_and(image>=thresh_otsu, label>0).astype(np.int8)
    region_2 = np.logical_and(image<thresh_otsu, label>0).astype(np.int8)

    new_label = np.zeros_like(label)
    new_label[region_1>0] = 2
    new_label[region_2>0] = 3

    # slice-wise
    boundaries, slice_indices = get_boundary(new_label, 2)
    print(f'Found valid {len(slice_indices)} slices:', slice_indices)

    margin_label = np.zeros_like(new_label)
    for bound, slice_idx in zip(boundaries, slice_indices):
        new_slice = np.zeros(region_1.shape[:2]).astype(np.int)
        for pt in bound:
            new_slice[pt[0], pt[1]] = 1

        #plt.imshow(new_slice, vmin=0, vmax=1)

        order = 3
        delta_dist = 5 # 5mm
        coords = np.array(bound).transpose()
        z = np.polyfit(coords[0], coords[1], order)
        p = np.poly1d(z)

        # # judge direction
        # pos_direction = False
        # center_idx = len(coords[0])//2
        # center_pt = (coords[0][center_idx], coords[1][center_idx])
        # center_delta = get_delta_xy(center_pt, reso, p, delta_dist)
        # center_pt_pos = (center_pt[0]+center_delta[0], center_pt[1]+center_delta[1])
        # if intersect(center_pt, center_pt_pos, region_2[...,slice_idx]):
        #     pos_direction = True
        # print('Positive direction:', pos_direction)
        
        new_line = []
        for x, y in zip(coords[0], coords[1]):
            delta_x_px, delta_y_px = get_delta_xy((x,y), reso, p, delta_dist)
            new_pt_pos = (x+delta_x_px, y+delta_y_px)
            new_pt_neg = (x-delta_x_px, y-delta_y_px)
            if intersect((x,y), new_pt_pos, region_2[...,slice_idx]):
                inter_pts = get_line_segment((x,y), new_pt_pos)
                new_line = new_line + inter_pts
            else:
                inter_pts = get_line_segment((x,y), new_pt_neg)
                new_line = new_line + inter_pts

        for pt in new_line:
            new_slice[pt[0], pt[1]] = out_label+8
        new_slice = binary_closing(new_slice, generate_binary_structure(2,1), iterations=2).astype(np.int)
        new_slice = binary_opening(new_slice,generate_binary_structure(2,1),iterations=1).astype(np.int)
        new_slice[new_slice>0] = out_label+8
        margin_label[..., slice_idx] = new_slice

    margin_label[new_label==2] = out_label
    #return margin_label
    nib.save(nib.Nifti1Image(margin_label, image_nii.affine, image_nii.header), 
             os.path.join(out_dir, os.path.basename(roi_path)))
    #nib.save(nib.Nifti1Image(new_label, image_nii.affine, image_nii.header), 
    #         os.path.join(out_dir, os.path.basename(roi_path).replace('.nii', '_new.nii')))

def shift_process(img_path, roi_path, out_label):
    image_nii = nib.load(img_path)
    label_nii = nib.load(roi_path)
    image = np.copy(image_nii.get_data())
    label = np.copy(label_nii.get_data())
    reso = image_nii.header['pixdim'][1:4]
    assert image.shape == label.shape, 'Image shape != Label shape'
    print('Image size:', image.shape, 'image reso:', reso)

    # Get threshold for boundary
    thresh_min = threshold_minimum(image[label>0])
    thresh_otsu = threshold_otsu(image[label>0])

    region_1 = np.logical_and(image>thresh_otsu, label>0).astype(np.int8)
    region_2 = np.logical_and(image<=thresh_otsu, label>0).astype(np.int8)

    new_label = np.zeros_like(label)
    new_label[region_1>0] = 2
    new_label[region_2>0] = 3

    # slice-wise
    boundaries, slice_indices = get_boundary(new_label, 2)
    print(f'Found valid {len(slice_indices)} slices:', slice_indices)

    margin_label = np.zeros_like(new_label)
    for bound, slice_idx in zip(boundaries, slice_indices):
        new_slice = np.zeros(region_1.shape[:2]).astype(np.int)
        for pt in bound:
            new_slice[pt[0], pt[1]] = 1

        order = 5
        delta_dist = 5 # 5mm
        coords = np.array(bound).transpose()
        z = np.polyfit(coords[0], coords[1], order)
        p = np.poly1d(z)

        # judge direction
        pos_direction = False
        center_idx = len(coords[0])//2
        center_pt = (coords[0][center_idx], coords[1][center_idx])
        center_delta = get_delta_xy(center_pt, reso, p, 10)
        center_pt_pos = (center_pt[0]+center_delta[0], center_pt[1]+center_delta[1])
        if intersect(center_pt, center_pt_pos, region_2[...,slice_idx]):
            pos_direction = True
        print('Positive direction:', pos_direction)
        
        new_line = []
        for x, y in zip(coords[0], coords[1]):
            delta_x_px, delta_y_px = get_delta_xy((x,y), reso, p, delta_dist, k=-1/p.deriv(1)(center_pt[0]))
            new_pt_pos = (x+delta_x_px, y+delta_y_px)
            new_pt_neg = (x-delta_x_px, y-delta_y_px)
            if pos_direction:
                inter_pts = get_line_segment((x,y), new_pt_pos)
                new_line = new_line + inter_pts
            else:
                inter_pts = get_line_segment((x,y), new_pt_neg)
                new_line = new_line + inter_pts

        for pt in new_line:
            new_slice[pt[0], pt[1]] = out_label+8
        new_slice = binary_closing(new_slice, generate_binary_structure(2,1), iterations=1).astype(np.int)
        new_slice[new_slice>0] = out_label+8
        margin_label[..., slice_idx] = new_slice

    margin_label[new_label==2] = out_label
    #return margin_label
    nib.save(nib.Nifti1Image(margin_label, image_nii.affine, image_nii.header), 
             os.path.join(out_dir, os.path.basename(roi_path)))

'''
#%%
if __name__ == '__main__bak__':
    # Read data and mask
    root_dir = r"E:\doctor tao\data\Ankle instability\DICOMDJW"
    out_dir  = os.path.join(root_dir, 'result')
    os.makedirs(out_dir, exist_ok=True)
    img_regx = 'src.nii'
    nii_paths = glob(os.path.join(root_dir, '*.nii'))

    img_paths = list(filter(lambda x: 'src.nii' in x, nii_paths))
    roi_paths = list(filter(lambda x: 'src.nii' not in x, nii_paths))
    assert len(img_paths)==1, 'No or multiple src image found!'
    assert len(roi_paths)>0, 'ROI not exists!'
    img_path = img_paths[0]

    CATEGORIES = {
        'L subtalar cal':1, 'L subtalar talus':2,
        'L talus dome':3, 'L tibial':4,
        'M subtalar cal':5, 'M subtalar talus':6,
        'M talus dome':7, 'M tibial':8
    }
    STRATEGY = 'norm' #'norm'
    # shift
    # STRATEGY = 'shift'

    image_nii = nib.load(img_path)

    prefix = os.path.basename(img_path).replace(img_regx,'')
    #final_mask = np.zeros(image_nii.shape)
    for roi_path in roi_paths:
        label_idx = CATEGORIES[os.path.basename(roi_path).replace(prefix,'').strip('.nii')]
        print('Processing label:', label_idx)

        if STRATEGY == 'norm':
            mask = norm_process(img_path, roi_path, label_idx)
            #final_mask = final_mask+mask
        elif STRATEGY == 'shift':
            mask = shift_process(img_path, roi_path, label_idx)

    # nib.save(nib.Nifti1Image(final_mask, image_nii.affine, image_nii.header), 
    #          os.path.join(out_dir, 'results.nii'))'''



# %% Padding program for knee 
from skimage.morphology import skeletonize
from matplotlib import pyplot as plt
import SimpleITK as sitk
from Tool.DataProcress import *

roi_fname = r"E:\Data\doctor_xie\knee_data\CAI NAI JI-20140605\roi.nii.gz"
roi = nib.load(roi_fname).get_fdata() # 必须使用这种方法

print('shape:', roi.shape, 'labels:', np.unique(roi))

for slice_idx in z_nonzero_slices(roi):
    img_slice = roi[...,slice_idx]
    unique_labels = np.unique(img_slice)
    candidates = np.intersect1d(unique_labels, [1,2,3,4])
    for target in candidates:
        # print(target)
        binary_slice = np.array(img_slice==target).astype(np.int8)
        # plt.imshow(binary_slice)
        # plt.show()
        skeleton = skeletonize(binary_slice).astype(np.int8)
        plt.imshow(skeleton)
        plt.show()

    #
        coord = list(zip(*np.where(skeleton>0)))
        endpts = list(filter(lambda x: end_points(skeleton, x), coord))
    #     print(f'Total {len(coord)} points, endpoints: {len(endpts)}')
    #     print(endpts)
        plt.imshow(binary_slice)
        plt.plot(np.array(endpts)[:, 1], np.array(endpts)[:, 0], color='cyan', marker='o',
                linestyle='None', markersize=6)
        plt.show()
    #
    #     fig, ax = plt.subplots()
    #     ax.imshow(binary_slice, cmap=plt.cm.gray)
    #     ax.plot(np.array(endpts)[:, 1], np.array(endpts)[:, 0], color='cyan', marker='o',
    #             linestyle='None', markersize=6)


# %%
