from IPython import display
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
from scipy.ndimage.morphology import binary_dilation, binary_closing
from scipy.ndimage import generate_binary_structure
from skimage.filters import threshold_minimum, threshold_otsu, threshold_local

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

#找到有ROI的层
def z_range(binary_image, axis=(0,1)):
    "returen non-zero z axis slices (min, max+1)"
    z = np.any(binary_image, axis=axis)
    zmin, zmax = np.where(z)[0][[0, -1]]
    return zmin, zmax+1


def get_boundary(binary_mask, region_label):
    boundaries = []
    slices = list(np.arange(*z_range(binary_mask)))
    for idx in slices:
        bin_slice = binary_mask[..., idx]
        temp = np.where(bin_slice==region_label)
        te = zip(*temp)
        coord = list(zip(*np.where(bin_slice==region_label))) 
        
        boundary = list(filter(lambda x: zero_neighbors(bin_slice, x), coord))
        boundaries.append(boundary)
    print(boundaries)
    return boundaries, slices

# judge whether line segment has intersection with target region
def intersect(p1, p2, region, nb_points=20, dilation=0):
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

def get_delta_xy(pt, reso, poly_fn, delta_dist):
    "Get x&y px-wise delta of given pt from specified curve fn"
    x = pt[0]
    # 求法向斜率，poly_fn.deriv()是关于x点求曲线的导数，相当于求x点的切线斜率
    k = -1 / poly_fn.deriv(1)(x) # norm direction
    #因为是沿着法向5mm，所以将其投影到x、y得到便宜分量
    delta_x = delta_dist / math.sqrt(1+k*k)
    delta_y = delta_dist / math.sqrt(1+k*k) * k
    delta_x_px = int(delta_x / reso[0])
    delta_y_px = int(delta_y / reso[1])
    return delta_x_px, delta_y_px


def process(img_path, roi_path, out_label):
    image_nii = nib.load(img_path)
    label_nii = nib.load(roi_path)
    image = np.copy(image_nii.get_data())
    label = np.copy(label_nii.get_data())
    reso = image_nii.header['pixdim'][1:4]
    assert image.shape == label.shape, 'Image shape != Label shape'
    # print('Image size:', image.shape, 'image reso:', reso)

    # Get threshold for boundary
    # thresh_min = threshold_minimum(image[label>0])
    #大津法
    thresh_otsu = threshold_otsu(image[label>0])
    
    # ax = plt.hist(image[label>0].ravel(), bins = 64)
    # plt.axvline(thresh_min, color='r')
    # plt.axvline(thresh_otsu, color='g')
    # plt.show()

    region_1 = np.logical_and(image>thresh_otsu, label>0).astype(np.int8)
    region_2 = np.logical_and(image<=thresh_otsu, label>0).astype(np.int8)

    new_label = np.zeros_like(label)
    #被大津法分割的两个区域分别赋值2、3，以脚踝为例：软骨区域为2，软骨下骨为3，因为软骨的信号是亮的，所以一定大于阈值
    new_label[region_1>0] = 2
    new_label[region_2>0] = 3

    # slice-wise
    # 得到软骨区域的外边沿
    boundaries, slice_indices = get_boundary(new_label, 2)
    # print(f'Found valid {len(slice_indices)} slices:', slice_indices)

    margin_label = np.zeros_like(new_label)
    for bound, slice_idx in zip(boundaries, slice_indices):
        new_slice = np.zeros(region_1.shape[:2]).astype(np.int)
        for pt in bound:
            new_slice[pt[0], pt[1]] = 1

        # plt.imshow(new_slice, vmin=0, vmax=1)
        # plt.show()

        order = 5
        delta_dist = 5 # 5mm
        #将pt再按[x1,x2...xn],[y1,y2...yn]
        coords = np.array(bound).transpose()
        # plt.imshow(bound)
        # plt.show()
        #用5次多项式拟合
        # print(coords)
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
            # 判段新的点是在外边缘点的基础上加还是减
            if intersect((x,y), new_pt_pos, region_2[...,slice_idx]):
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
    # nib.save(nib.Nifti1Image(new_label, image_nii.affine, image_nii.header),
    #         os.path.join(out_dir, os.path.basename(roi_path).replace('.nii', '_new.nii')))



if __name__ == '__main__':
    # Read data and mask
    def run(case_path):
        root_dir = case_path
        global out_dir
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

        image_nii = nib.load(img_path)

        prefix = os.path.basename(img_path).replace(img_regx,'')
        #final_mask = np.zeros(image_nii.shape)
        for roi_path in roi_paths:
            try:
                label_idx = CATEGORIES[os.path.basename(roi_path).replace(prefix,'').strip('.nii')]
                # print('Processing label:', label_idx)

                mask = process(img_path, roi_path, label_idx)
                #final_mask = final_mask+mask
            except:
                label_idx = CATEGORIES[os.path.basename(roi_path).split("_")[-1].split(".")[0]]
                mask = process(img_path, roi_path, label_idx)


        # nib.save(nib.Nifti1Image(final_mask, image_nii.affine, image_nii.header),
        #          os.path.join(out_dir, 'results.nii'))



        # %%

    all_data_path = r'H:\tao_20200715'
    for folder in os.listdir(all_data_path):
        folder_path = os.path.join(all_data_path, folder)
        # for case in os.listdir(folder_path):
        #     case_path = os.path.join(folder_path, case)
            # run(case_path)
        # print(folder_path)
        # try:
        run(case_path=folder_path)
        # except:
        #     print(folder_path)
        # break


