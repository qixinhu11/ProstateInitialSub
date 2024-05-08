import os, shutil
import numpy as np 
import nibabel as nib 
import os, json, monai, torch
import numpy as np 
import nibabel as nib
import matplotlib.pyplot as plt
import cv2 as cv2
import SimpleITK as sitk
from skimage.measure import label  
from monai import transforms, data
from scipy import ndimage

# classification preprocessing
def _get_transform():
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["T2W", "DWI", "ADC", "label"]),
            transforms.EnsureChannelFirstd(keys=["T2W", "DWI", "ADC", "label"]),
            transforms.Orientationd(keys=["T2W", "DWI", "ADC", "label"], axcodes="RAI"),
            transforms.Spacingd(keys=["T2W", "DWI", "ADC", "label"], pixdim=(0.5, 0.5, 3), mode=("bilinear","bilinear", "bilinear", "nearest")),
            transforms.ToTensord(keys=["T2W", "DWI", "ADC", "label"]),
        ]
        )
    return train_transform

def find_bbox_center(seg):
    x_start, x_end = np.where(np.any(seg, axis=(1, 2)))[0][[0, -1]]
    y_start, y_end = np.where(np.any(seg, axis=(0, 2)))[0][[0, -1]]
    z_start, z_end = np.where(np.any(seg, axis=(0, 1)))[0][[0, -1]]

    x_center = (x_start + x_end) // 2
    y_center = (y_start + y_end) // 2
    # z_center = (z_start + z_end) // 2
    
    x_size, y_size = 64, 64

    x_start, x_end = max(0, x_center-x_size), min(seg.shape[0], x_center+x_size)
    y_start, y_end = max(0, y_center-y_size), min(seg.shape[1], y_center+y_size)
    
    # x_start, x_end = max(0, x_start-5), min(seg.shape[0], x_end+5)
    # y_start, y_end = max(0, y_start-5), min(seg.shape[1], y_end+5)
    z_start, z_end = max(0, z_start-1), min(seg.shape[2], z_end+1)

    return (x_start, x_end, y_start, y_end, z_start, z_end)

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def keep_largest_component(segmentation, labels):
    mask = np.zeros_like(segmentation, dtype=np.uint8) # don't change the input
    for l in labels:
        segmentation_l = (segmentation == l)
        segmentation_l_LCC = getLargestCC(segmentation_l)
        mask[segmentation_l_LCC == True] = l 
    return mask

def preprocessing_cls(save_root, datalist):

    transform = _get_transform()
    ds = data.Dataset(data=datalist, transform=transform)
    norm_function = transforms.NormalizeIntensity(nonzero=True, channel_wise=True)

    for item in ds:
        adc = item['ADC'][0]
        t2w = item['T2W'][0]
        dwi = item['DWI'][0]
        mask = item['label'][0]

        mask = keep_largest_component(mask, [1])
        x_start, x_end, y_start, y_end, z_start, z_end = find_bbox_center(mask)


        crop_adc  = adc[x_start:x_end, y_start:y_end, z_start:z_end]
        crop_t2w  = t2w[x_start:x_end, y_start:y_end, z_start:z_end]
        crop_dwi  = dwi[x_start:x_end, y_start:y_end, z_start:z_end]
        crop_mask = mask[x_start:x_end, y_start:y_end, z_start:z_end]   

        crop_adc = norm_function(crop_adc)
        crop_t2w = norm_function(crop_t2w)
        crop_dwi = norm_function(crop_dwi)
        

        crop_adc = ndimage.zoom(crop_adc,zoom=(1, 1, 12/crop_adc.shape[-1]))
        crop_t2w = ndimage.zoom(crop_t2w,zoom=(1, 1, 12/crop_t2w.shape[-1]))
        crop_dwi = ndimage.zoom(crop_dwi,zoom=(1, 1, 12/crop_dwi.shape[-1]))
        crop_mask  = ndimage.zoom(crop_mask,zoom=(1, 1, 12/crop_mask.shape[-1]), mode="nearest")

        crop_img = np.concatenate([crop_t2w, crop_dwi, crop_adc], axis=-1)
        print(item['T2W_meta_dict']['filename_or_obj'], crop_img.shape, np.max(crop_mask), (np.max(crop_adc), np.max(crop_t2w), np.max(crop_dwi)), (crop_adc.shape, crop_t2w.shape, crop_dwi.shape))
        name = item['T2W_meta_dict']['filename_or_obj'].split('/')[-2].split('_')[0]
        np.savez(os.path.join(save_root, name+'.npz'), img=crop_img, mask=crop_mask)

    prefix = '/'.join(save_root.split('/')[:-1])
    T2W = os.path.join(prefix,"T2W")
    DWI = os.path.join(prefix,"DWI")
    ADC = os.path.join(prefix,"ADC")
    T2W_DWI = os.path.join(prefix,"T2W_DWI")
    T2W_ADC = os.path.join(prefix,"T2W_ADC")
    DWI_ADC = os.path.join(prefix,"DWI_ADC")

    os.makedirs(T2W,exist_ok=True)
    os.makedirs(DWI,exist_ok=True)
    os.makedirs(ADC,exist_ok=True)
    os.makedirs(T2W_DWI,exist_ok=True)
    os.makedirs(T2W_ADC,exist_ok=True)
    os.makedirs(DWI_ADC,exist_ok=True)

    name_list = sorted(os.listdir(save_root))
    for name in name_list:
        item = np.load(os.path.join(save_root, name))
        img = item['img']
        single_mask = item['mask'][...,:12]
        two_mask = item['mask'][...,:24]

        t2w = img[...,:12]
        dwi = img[...,12:24]
        adc = img[...,24:36]

        t2w_dwi = np.concatenate([t2w, dwi], axis=-1)
        t2w_adc = np.concatenate([t2w, adc], axis=-1)
        dwi_adc = np.concatenate([dwi, adc], axis=-1)

        # save the different modalitys
        np.savez(os.path.join(T2W, name), img=t2w, mask=single_mask)
        np.savez(os.path.join(DWI, name), img=dwi, mask=single_mask)
        np.savez(os.path.join(ADC, name), img=adc, mask=single_mask)

        np.savez(os.path.join(T2W_DWI, name), img=t2w_dwi, mask=two_mask)
        np.savez(os.path.join(T2W_ADC, name), img=t2w_adc, mask=two_mask)
        np.savez(os.path.join(DWI_ADC, name), img=dwi_adc, mask=two_mask)
        print(name, 'is move to all modality')

def cls_processing(aligned_root, preprocessing_root):
    label_root = os.path.join(preprocessing_root, 'CLS')
    preprocessing_root = os.path.join(preprocessing_root, 'CLS', 'ALL')
    os.makedirs(preprocessing_root, exist_ok=True)

    # move current file to the classification root
    shutil.copy2('./test_case_level.txt', os.path.join(label_root, "test_case_level.txt"))
    shutil.copy2('./train_case_level.txt', os.path.join(label_root, "train_case_level.txt"))
    shutil.copy2('./case_level_label.npz', os.path.join(label_root, "case_level_label.npz"))

    print("="*20, end='\t')
    print("Start to do pre-processing for classification")
    name_list = os.listdir(aligned_root)
    datalist = []
    for name in name_list:
        sample = {"label": os.path.join(aligned_root, name, f'{name}_T2W_gt.nii.gz')}
        for modality in ["T2W", "ADC", "DWI"]:
            sample[modality] = os.path.join(aligned_root, name, f'{name}_{modality}.nii.gz')
        datalist.append(sample)
    preprocessing_cls(preprocessing_root, datalist)

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
     
def get_single_case(aligned_root, name):
    t2w = sitk.ReadImage(os.path.join(aligned_root, name, name+ "_T2W.nii.gz"))
    t2w = sitk.GetArrayFromImage(t2w)
    t2w = normalize(t2w)
    t2w = np.transpose(t2w, (1, 2, 0))

    dwi = sitk.ReadImage(os.path.join(aligned_root, name, name+ "_DWI.nii.gz"))
    dwi = sitk.GetArrayFromImage(dwi)
    dwi = normalize(dwi)
    dwi = np.transpose(dwi, (1, 2, 0))

    adc = sitk.ReadImage(os.path.join(aligned_root, name, name+ "_ADC.nii.gz"))
    adc = sitk.GetArrayFromImage(adc)
    adc = normalize(adc)
    adc = np.transpose(adc, (1, 2, 0))

    t2w_gt = sitk.ReadImage(os.path.join(aligned_root, name, name+ "_T2W_gt.nii.gz"))
    t2w_gt = sitk.GetArrayFromImage(t2w_gt)
    t2w_gt_old = np.transpose(t2w_gt, (1, 2, 0))
    t2w_gt = np.zeros_like(t2w_gt_old)
    t2w_gt[t2w_gt_old>1] = 1

    dwi_gt = sitk.ReadImage(os.path.join(aligned_root, name, name+ "_DWI_gt.nii.gz"))
    dwi_gt = sitk.GetArrayFromImage(dwi_gt)
    dwi_gt_old = np.transpose(dwi_gt, (1, 2, 0))
    dwi_gt = np.zeros_like(dwi_gt_old)
    dwi_gt[dwi_gt_old>1] = 1

    adc_gt = sitk.ReadImage(os.path.join(aligned_root, name, name+ "_ADC_gt.nii.gz"))
    adc_gt = sitk.GetArrayFromImage(adc_gt)
    adc_gt_old = np.transpose(adc_gt, (1, 2, 0))
    adc_gt = np.zeros_like(adc_gt_old)
    adc_gt[adc_gt_old>1] = 1

    # resize to 224*224*30
    x,y,z = t2w.shape
    resize_x,resize_y,resize_z = 224, 224, 30
    zoom_t2w = ndimage.zoom(t2w, zoom=(resize_x/x, resize_y/y, resize_z/z))
    zoom_adc = ndimage.zoom(adc, zoom=(resize_x/x, resize_y/y, resize_z/z))
    zoom_dwi = ndimage.zoom(dwi, zoom=(resize_x/x, resize_y/y, resize_z/z))

    zoom_t2w_gt = ndimage.zoom(t2w_gt, zoom=(resize_x/x, resize_y/y, resize_z/z), mode="nearest")
    zoom_adc_gt = ndimage.zoom(adc_gt, zoom=(resize_x/x, resize_y/y, resize_z/z), mode="nearest")
    zoom_dwi_gt = ndimage.zoom(dwi_gt, zoom=(resize_x/x, resize_y/y, resize_z/z), mode="nearest")

    return zoom_t2w, zoom_adc, zoom_dwi, zoom_t2w_gt, zoom_adc_gt, zoom_dwi_gt

def seg_processing(aligned_root, preprocessing_root):
    preprocessing_root = os.path.join(preprocessing_root, 'SEG')
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

    parser.add_argument('--tingting_save_path', default="/Users/qixinhu/Project/CUHK/Prostate/PAIsData/0426/tingting", type=str)
    parser.add_argument('--qixin_save_path', default="/Users/qixinhu/Project/CUHK/Prostate/PAIsData/0426/qixin", type=str)

    args = parser.parse_args()

    # cls_processing(args.tingting_save_path, args.qixin_save_path)
    seg_processing(args.tingting_save_path, args.qixin_save_path)