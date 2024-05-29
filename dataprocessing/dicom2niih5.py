import SimpleITK as sitk
import os, sys
import numpy as np
import cv2
import copy
import h5py

def norm(im):
    # mean_ = np.mean(im)
    # std_ = np.std(im)
    # im = (im - mean_) / std_
    max_ = np.max(im)
    min_ = np.min(im)
    im  = (im - min_) / (max_ -  min_)
    im = (im - 0.5)*2
    return im
def read_dcm(dcm_directory):
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dcm_directory)
    if not series_ids:
        print("ERROR: given directory dose not a DICOM series.")
        sys.exit(1)
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dcm_directory,series_ids[0])
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    image3D = series_reader.Execute()
    return image3D
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
def align_seg_with_raw_nrrd(dcm, seg):
    # Just for labelmap .... because of nearestNeighour interpolator
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(dcm)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    seg_new = resampler.Execute(seg)
    return seg_new

def wenao_save(T2W_images, DWI_images, ADC_images, save_path):
    # T2W
    image3d_T2W = T2W_images
    Images = sitk.GetArrayFromImage(image3d_T2W)
    originImg = np.zeros((Images.shape[0],224,224)).astype(np.float32)
    Images = normalize(Images)
    resultImage_ = sitk.GetImageFromArray(Images)  
    sitk.WriteImage(resultImage_, os.path.join(save_path,'T2W_raw.nii.gz'))

    for i in range(len(Images)):
        originImg[i] = cv2.resize(Images[i], (224, 224))
    resultImage_ = sitk.GetImageFromArray(originImg)  
    sitk.WriteImage(resultImage_, os.path.join(save_path,'T2W.nii.gz'))

    # DWI
    image3d = DWI_images
    sitk.WriteImage(image3d, save_path+'DWI_raw.nii.gz')
    image3d = align_seg_with_raw_nrrd(image3d_T2W, image3d)
    Images = sitk.GetArrayFromImage(image3d)
    originImg = np.zeros((Images.shape[0],224,224)).astype(np.float32)
    Images = normalize(Images)

    for i in range(len(Images)):
        originImg[i] = cv2.resize(Images[i], (224, 224))
    resultImage_ = sitk.GetImageFromArray(originImg)  
    sitk.WriteImage(resultImage_, save_path+'DWI.nii.gz')

    # ADC
    image3d = ADC_images
    sitk.WriteImage(image3d, save_path+'ADC_raw.nii.gz')
    image3d = align_seg_with_raw_nrrd(image3d_T2W, image3d)
    Images = sitk.GetArrayFromImage(image3d)
    originImg = np.zeros((Images.shape[0],224,224)).astype(np.float32)
    Images = normalize(Images)

    for i in range(len(Images)):
        originImg[i] = cv2.resize(Images[i], (224, 224))
    resultImage_ = sitk.GetImageFromArray(originImg)  
    sitk.WriteImage(resultImage_, save_path+'ADC.nii.gz')
def tiantian_save(T2W_images, DWI_images, ADC_images, T2W_gt, DWI_gt, ADC_gt, save_root, name):

    # T2W
    os.makedirs(os.path.join(save_root, name), exist_ok=True)
    T2W_save_name = os.path.join(save_root, name, name+"_T2W")
    sitk.WriteImage(T2W_images, T2W_save_name+'.nii.gz')
    
    T2W_gt = sitk.ReadImage(T2W_gt)
    sitk.WriteImage(T2W_gt, T2W_save_name+'_gt.nii.gz')

    # DWI
    DWI_images = align_seg_with_raw_nrrd(T2W_images, DWI_images)
    DWI_save_name = os.path.join(save_root, name, name+"_DWI")
    sitk.WriteImage(DWI_images, DWI_save_name+'.nii.gz')
    
    DWI_gt = sitk.ReadImage(DWI_gt)
    DWI_gt = align_seg_with_raw_nrrd(T2W_gt, DWI_gt)
    sitk.WriteImage(DWI_gt, DWI_save_name+'_gt.nii.gz')

    # ADC
    ADC_images = align_seg_with_raw_nrrd(T2W_images, ADC_images)
    ADC_save_name = os.path.join(save_root, name, name+"_ADC")
    sitk.WriteImage(ADC_images, ADC_save_name+'.nii.gz')

    ADC_gt = sitk.ReadImage(ADC_gt)
    ADC_gt = align_seg_with_raw_nrrd(T2W_gt, ADC_gt)
    sitk.WriteImage(ADC_gt, ADC_save_name+'_gt.nii.gz')



def main(raw_dicom_root, tiantian_save_root, wenao_save_root):
    # Dicon to .nii.gz
    print("="*20, end='\t')
    print("Start to transfer .dicon to .nii.gz file")

    path_list = os.listdir(raw_dicom_root)
    for dir in path_list:
        name = dir.split('_')[0]

        wenao_save_path = wenao_save_root + '/' + dir.split('_')[0] + '/'
        os.makedirs(wenao_save_path,exist_ok=True)

        T2W_path = os.path.join(raw_dicom_root, dir, name + '_T2W/')
        DWI_path = os.path.join(raw_dicom_root, dir, name + '_DWI')
        ADC_path = os.path.join(raw_dicom_root, dir, name + '_ADC/')

        T2W_images = read_dcm(T2W_path)
        DWI_images = read_dcm(DWI_path)
        ADC_images = read_dcm(ADC_path)

        T2W_gt = os.path.join(raw_dicom_root, dir, name + '_T2W/', name + '_T2W.nii.gz')
        DWI_gt = os.path.join(raw_dicom_root, dir, name + '_DWI/', name + '_DWI.nii.gz')
        ADC_gt = os.path.join(raw_dicom_root, dir, name + '_ADC/', name + '_ADC.nii.gz')

        # wenao save T2W/DWI/ADC ways
        wenao_save(T2W_images, DWI_images, ADC_images, wenao_save_path)
        # tiantian save T2W/DWI/ADC ways
        tiantian_save(T2W_images, DWI_images, ADC_images, T2W_gt, DWI_gt, ADC_gt, tiantian_save_root, name)

        print(name, "done")

    # .nii.gz to .h5 file
    print("="*20, end='\t')
    print("Start to transfer .nii.gz to .h5 file")
    path_list = os.listdir(wenao_save_root)
    try:
        os.remove(r'./test_PZ.txt')
    except:
        print("")

    for idx in range(len(path_list)):
        slice_name = path_list[idx].split('_')[0]
        img_path = os.path.join(wenao_save_root, slice_name, "T2W.nii.gz")
        img_sitk = sitk.ReadImage(img_path)
        img_T2W = sitk.GetArrayFromImage(img_sitk)

        img_path = os.path.join(wenao_save_root, slice_name, "DWI.nii.gz")
        img_sitk = sitk.ReadImage(img_path)
        img_DWI = sitk.GetArrayFromImage(img_sitk)
        
        img_path = os.path.join(wenao_save_root, slice_name, "ADC.nii.gz")
        img_sitk = sitk.ReadImage(img_path)
        img_ADC = sitk.GetArrayFromImage(img_sitk)

        gt = np.zeros(img_ADC.shape)
        DWI_gt = np.zeros(img_ADC.shape)
        ADC_gt = np.zeros(img_ADC.shape)
        Con_gt = np.zeros(img_ADC.shape)

        img_resize = np.zeros((img_T2W.shape[0],3,224,224))
        gt_resize = np.zeros((img_T2W.shape[0],5,224,224))

        for num in range(len(img_T2W)):
            img_resize[num,0] = img_T2W[num]
            img_resize[num,1] = img_DWI[num]
            img_resize[num,2] = img_ADC[num]
            gt_resize[num,0] = gt[num]
            gt_resize[num,1] = DWI_gt[num]
            gt_resize[num,2] = ADC_gt[num]
            gt_resize[num,3] = Con_gt[num]
        
        f = h5py.File(wenao_save_root + '/' + slice_name +'/data_con_PZ.npy.h5','w')
        f['image'] = img_resize
        f['label'] = gt_resize
        f.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='dicom to nii file')

    parser.add_argument('--test_path', default="/Users/qixinhu/Project/CUHK/Prostate/PAIsData/PAIs_Data2AIteam_20240426/Case_lesion12345/Test", type=str)
    parser.add_argument('--train_path', default="/Users/qixinhu/Project/CUHK/Prostate/PAIsData/PAIs_Data2AIteam_20240426/Case_lesion12345/Train", type=str)
    parser.add_argument('--tiantian_save_path', default="/Users/qixinhu/Project/CUHK/Prostate/PAIsData/0426/tiantian", type=str)
    parser.add_argument('--wenao_save_path', default="/Users/qixinhu/Project/CUHK/Prostate/PAIsData/0426/wenao", type=str)
    args = parser.parse_args()

    main(args.test_path, args.tiantian_save_path, args.wenao_save_path)
    main(args.train_path, args.tiantian_save_path, args.wenao_save_path)
    print("="*20, end='\t')
    print("All done!")