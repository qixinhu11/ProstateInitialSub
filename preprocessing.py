import os 
import numpy as np 
import nibabel as nib 
import os, json, monai, torch
import numpy as np 
import nibabel as nib
import matplotlib.pyplot as plt
import cv2 as cv2
from skimage.measure import label  
from monai import transforms, data
from scipy import ndimage

# change the name and the spacing
def change_spacing(prediction_root, aligned_root):
    aligned_root = 'data/raw_aligned/'
    prediction_root = 'data/all_1113/'
    case_list = os.listdir(aligned_root)
    for case in case_list:
        t2w_case = nib.load(os.path.join(aligned_root, case, case+"_T2W.nii.gz"))
        pred_mask = nib.load(os.path.join(prediction_root, case, "T2W_PZ_raw.nii.gz" )).get_fdata()
        original_affine  = t2w_case.affine
        save_name = os.path.join(aligned_root, case, case+"_T2W_gt.nii.gz")
        nib.save(
            nib.Nifti1Image(pred_mask.astype(np.uint8), original_affine), save_name
        )
        print(case, 'done')
    

# do classification preprocessing
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


def preprocessing(save_root, datalist):

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


if __name__ == "__main__":
    prediction_root = 'data/all_1113/'
    aligned_root = 'data/raw_aligned/'
    preprocessing_root = 'data/preprocess/ALL'
    os.makedirs(preprocessing_root, exist_ok=True)

    print("="*20, end='\t')
    print("Start to change the spacing of precition segmentation")
    change_spacing(prediction_root, aligned_root)


    print("="*20, end='\t')
    print("Start to do pre-processing for classification")
    name_list = os.listdir(aligned_root)
    datalist = []
    for name in name_list:
        sample = {"label": os.path.join(aligned_root, name, f'{name}_T2W_gt.nii.gz')}
        for modality in ["T2W", "ADC", "DWI"]:
            sample[modality] = os.path.join(aligned_root, name, f'{name}_{modality}.nii.gz')
        datalist.append(sample)

    preprocessing(preprocessing_root, datalist)
    print("="*20, end='\t')
    print("All done!")

