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

    label_root = args.file_root

    test_list = []
    with open('./cspca_test.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            test_list.append(line)

    dice_list = []
    for item in test_list:
        # pred0 = np.load(os.path.join(pred_root0, item+'.npz'))['pred'] # 224*224*30
        pred1 = np.load(os.path.join(pred_root1, item+'.npz'))['pred'] # 224*224*30
        pred2 = np.load(os.path.join(pred_root2, item+'.npz'))['pred'] # 224*224*30
        pred3 = np.load(os.path.join(pred_root3, item+'.npz'))['pred'] # 224*224*30
        pred = (( pred3) > 0).astype(np.float32) # 224*224*30
        label = np.load(os.path.join(label_root, item+'.npz'))['mask'][args.modality,...].astype(np.float32) # 224*224*30

        current_dice = cal_dice(pred, label)
        dice_list.append(current_dice)
        print(item, current_dice)
    print(np.mean(dice_list))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--modality', default=0, type=int)
    parser.add_argument('--file_root', default="/Users/qixinhu/Project/CUHK/Prostate/PAIsData/0426/qixin/SEG", type=str)

    args = parser.parse_args()

    get_dice(args)