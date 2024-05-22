import os 
import numpy as np 

def cal_dice(pred, label):
    intersection = np.logical_and(pred, label).sum()
    total_pixels = pred.sum() + label.sum()
    dice_score = (2.0 * intersection) / (1e-9 + total_pixels)
    return dice_score


def get_dice(args):
    pred_root0 = "out/BS1_LR0.0001_MOD0"
    pred_root1 = "out/BS1_LR0.0001_MOD1"
    pred_root2 = "out/BS1_LR0.0001_MOD2"
    pred_root3 = "out/BS1_LR0.0001_MOD3"
    real_moe_root = "out/MoE_BS1_LR0.0001/"

    label_root = args.file_root

    test_list = []
    with open('./cspca_test.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            test_list.append(line)

    list_self = []
    list_01 = []
    list_02 = []
    list_03 = []
    list_12 = []
    list_13 = []
    list_23 = []
    list_012 = []
    list_013 = []
    list_123 = []
    list_0123 = []
    list_moe = []

    for item in test_list:
        pred0 = np.load(os.path.join(pred_root0, item+'.npz'))['pred'] # 224*224*30
        pred1 = np.load(os.path.join(pred_root1, item+'.npz'))['pred'] # 224*224*30
        pred2 = np.load(os.path.join(pred_root2, item+'.npz'))['pred'] # 224*224*30
        pred3 = np.load(os.path.join(pred_root3, item+'.npz'))['pred'] # 224*224*30
        pred_moe = np.load(os.path.join(real_moe_root, item+'.npz'))['pred']

        # pred = ((pred1) > 0).astype(np.float32) # 224*224*30
        label = np.load(os.path.join(label_root, item+'.npz'))['mask'][args.modality,...].astype(np.float32) # 224*224*30
        if args.modality == 0:
            pred_self = pred0 
        elif args.modality == 1:
            pred_self = pred1 
        elif args.modality == 2:
            pred_self = pred2
        
        pred_01 =  ((pred0 + pred1) > 0).astype(np.float32)
        pred_02 =  ((pred0 + pred2) > 0).astype(np.float32)
        pred_03 =  ((pred0 + pred3) > 0).astype(np.float32)
        pred_13 =  ((pred1 + pred3) > 0).astype(np.float32)
        pred_23 =  ((pred2 + pred3) > 0).astype(np.float32)
        pred_12 =  ((pred1 + pred2) > 0).astype(np.float32)
        pred_012 = ((pred0 + pred1 + pred2) > 0).astype(np.float32)
        pred_013 = ((pred0 + pred1 + pred3) > 0).astype(np.float32)
        pred_123 = ((pred1 + pred2 + pred3) > 0).astype(np.float32)
        pred_0123 = ((pred0 + pred1 + pred2 + pred3) > 0).astype(np.float32)


        dice_self = cal_dice(pred_self, label)
        dice_01 = cal_dice(pred_01, label)
        dice_02 = cal_dice(pred_02, label)
        dice_03 = cal_dice(pred_03, label)
        dice_12 = cal_dice(pred_12, label)
        dice_13 = cal_dice(pred_13, label)
        dice_23 = cal_dice(pred_23, label)
        dice_012 = cal_dice(pred_012, label)
        dice_013 = cal_dice(pred_013, label)
        dice_123 = cal_dice(pred_123, label)
        dice_0123 = cal_dice(pred_0123, label)
        dice_moe = cal_dice(pred_moe, label)

        list_self.append(dice_self)
        list_01.append(dice_01)
        list_02.append(dice_02)
        list_03.append(dice_03)
        list_12.append(dice_12)
        list_13.append(dice_13)
        list_23.append(dice_23)
        list_012.append(dice_012)
        list_013.append(dice_013)
        list_123.append(dice_123)
        list_0123.append(dice_0123)
        list_moe.append(dice_moe)

        print(item + "||", dice_self, dice_01, dice_02, dice_03, dice_12, dice_13, dice_23, dice_012, dice_013, dice_123, dice_0123, dice_moe)
    print("{:.3f} || {:.3f} || {:.3f} || {:.3f} || {:.3f} || {:.3f} || {:.3f} || {:.3f} || {:.3f} || {:.3f}".format(
        np.mean(list_01), np.mean(list_02), np.mean(list_03), np.mean(list_12), np.mean(list_13),
        np.mean(list_23), np.mean(list_012), np.mean(list_013), np.mean(list_123) ,np.mean(list_0123)
    ))
    print("No MoE:", np.mean(list_self))
    print("Read MoE:", np.mean(list_moe))
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--modality', default=0, type=int)
    parser.add_argument('--file_root', default="/Users/qixinhu/Project/CUHK/Prostate/PAIsData/0426/qixin/SEG", type=str)

    args = parser.parse_args()

    get_dice(args)