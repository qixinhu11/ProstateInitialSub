import os, shutil
import numpy as np 
import SimpleITK as sitk
from scipy.ndimage import label
from scipy import ndimage

# segmentation preprocessing
def normalize(img_):
    #img_[img_>=250] = 250
    #img_[img_<=-200] = -200
    mean = np.mean(img_)
    std = np.std(img_)
    # mean = -0.29790374886599763
    # std = 0.29469745653088375
    img_ = (img_ - mean) / std
    max_ = np.max(img_)
    min_ = np.min(img_)
    img_ = (img_ - min_) / (max_ - min_ + 1e-9)
    img_ = img_ * 2 - 1
    return img_

def read_nii(name):
    data = sitk.ReadImage(name)
    data = sitk.GetArrayFromImage(data)
    data = np.transpose(data, (1, 2, 0))
    return data

def round2mask(t2w_gt):
    new_t2w = (t2w_gt==0) + (t2w_gt>1)
    mask = np.zeros_like(new_t2w)
    for z in range(new_t2w.shape[-1]):
        labels, nb = label(new_t2w[...,z])
        for idx in range(1, nb+1):
            if idx == 2:
                mask[...,z] = (labels == idx)
    mask = ndimage.binary_dilation(mask, iterations=1)
    return mask

def get_bbox(seg:np.ndarray):
    x_start, x_end = np.where(np.any(seg, axis=(1, 2)))[0][[0, -1]]
    y_start, y_end = np.where(np.any(seg, axis=(0, 2)))[0][[0, -1]]
    # z_start, z_end = np.where(np.any(seg, axis=(0, 1)))[0][[0, -1]]

    x_center = (x_start + x_end) // 2
    y_center = (y_start + y_end) // 2
    # z_center = (z_start + z_end) // 2

    return x_center, y_center

def get_zoom_image(org_image:np.ndarray, centers, widths, mode="constant", order=3):
    """
    zoom image in into 224 * 224 * z
    """
    x_start, x_end = max(0,centers[0]-widths[0]), min(org_image.shape[0], centers[0]+widths[0])
    y_start, y_end = max(0,centers[1]-widths[1]), min(org_image.shape[1], centers[1]+widths[1])

    crop_image = org_image[x_start:x_end,y_start:y_end,...]
    x,y,_ = crop_image.shape
    zoom_image = ndimage.zoom(crop_image,zoom=(224/x, 224/y, 1),mode=mode,order=order)
    return zoom_image


def get_single_case(aligned_root, name):
    t2w_gt_old = read_nii(os.path.join(aligned_root, name, name+ "_T2W_gt.nii.gz"))
    prostate_mask = round2mask(t2w_gt_old)
    t2w_gt = np.zeros_like(t2w_gt_old)
    t2w_gt[prostate_mask==1] = 1
    t2w_gt[t2w_gt_old>1] = 2

    dwi_gt_old = read_nii(os.path.join(aligned_root, name, name+ "_DWI_gt.nii.gz"))
    dwi_gt = np.zeros_like(dwi_gt_old)
    dwi_gt[prostate_mask==1] = 1
    dwi_gt[dwi_gt_old>1] = 2

    adc_gt_old = read_nii(os.path.join(aligned_root, name, name+ "_ADC_gt.nii.gz"))
    adc_gt = np.zeros_like(adc_gt_old)
    adc_gt[prostate_mask==1] = 1
    adc_gt[adc_gt_old>1] = 2

    t2w = read_nii(os.path.join(aligned_root, name, name+ "_T2W.nii.gz"))
    dwi = read_nii(os.path.join(aligned_root, name, name+ "_DWI.nii.gz"))
    adc = read_nii(os.path.join(aligned_root, name, name+ "_ADC.nii.gz"))
    t2w = normalize(t2w)
    dwi = normalize(dwi)
    adc = normalize(adc)
    original_x, original_y, original_z = t2w.shape
    ROI_ratio = 0.6
    widths  = (int(original_x * ROI_ratio) // 2, int(original_y * ROI_ratio) // 2)
    centers = get_bbox(t2w_gt_old)

    zoom_t2w = get_zoom_image(t2w, centers, widths)
    zoom_dwi = get_zoom_image(dwi, centers, widths)
    zoom_adc = get_zoom_image(adc, centers, widths)

    zoom_t2w_gt = get_zoom_image(t2w_gt, centers, widths, mode="nearest", order=0)
    zoom_adc_gt = get_zoom_image(adc_gt, centers, widths, mode="nearest", order=0)
    zoom_dwi_gt = get_zoom_image(dwi_gt, centers, widths, mode="nearest", order=0)

    return zoom_t2w, zoom_adc, zoom_dwi, zoom_t2w_gt, zoom_adc_gt, zoom_dwi_gt

def seg_processing(aligned_root, preprocessing_root):
    preprocessing_root = os.path.join(preprocessing_root, 'SEGv2')
    os.makedirs(preprocessing_root, exist_ok=True)
    shutil.copy2('./test_case_level.txt', os.path.join(preprocessing_root, "test_case_level.txt"))
    shutil.copy2('./train_case_level.txt', os.path.join(preprocessing_root, "train_case_level.txt"))
    shutil.copy2('./case_level_label.npz', os.path.join(preprocessing_root, "case_level_label.npz"))
    name_list = os.listdir(aligned_root)
    print("="*20, end='\t')
    print("Start to do pre-processing for segmentation, total of {} cases.".format(len(name_list)))
    for idx, name in enumerate(name_list):
        zoom_t2w, zoom_adc, zoom_dwi, zoom_t2w_gt, zoom_adc_gt, zoom_dwi_gt = get_single_case(aligned_root, name)
        img = np.stack([zoom_t2w, zoom_dwi, zoom_adc], axis=0)
        mask = np.stack([zoom_t2w_gt, zoom_dwi_gt, zoom_adc_gt], axis=0)

        np.savez(os.path.join(preprocessing_root, name), img=img, mask=mask)
        print(f"{idx+1}/{len(name_list)} ||", name, img.shape, mask.shape, 'done')
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='dicom to nii file')

    parser.add_argument('--tiantian_save_path', default="/Users/qixinhu/Project/CUHK/Prostate/PAIsData/0426/tiantian", type=str)
    parser.add_argument('--qixin_save_path', default="/Users/qixinhu/Project/CUHK/Prostate/PAIsData/0426/qixin", type=str)

    args = parser.parse_args()

    seg_processing(args.tiantian_save_path, args.qixin_save_path)