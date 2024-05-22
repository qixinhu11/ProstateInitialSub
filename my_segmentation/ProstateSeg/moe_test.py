import os, json, torch, monai
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from monai.inferers import sliding_window_inference
from monai.data import DataLoader, decollate_batch
from monai import transforms
from dataset.prostateMoE import prostateMoE
import warnings
warnings.filterwarnings("ignore")

def _get_model(args):
    from models.MoESeg import MoESeg
    model = MoESeg()
    model_path = f"out/MoE_BS1_LR0.0001/model_{args.epoch}.pth"
    model.load_state_dict(torch.load(model_path)['net'])
    model.to(args.device)
    return model
    
def _get_loader(args):
    file_root = args.file_root
    test_samples = []
    with open(os.path.join(file_root,'test_case_level.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            sample = os.path.join(file_root, line.strip() + '.npz')
            test_samples.append(sample)

    val_img_transform = transforms.Compose(
        [
            transforms.EnsureChannelFirst(channel_dim=0),
        ]
    )
    val_seg_transform = transforms.Compose(
        [
            transforms.EnsureChannelFirst(channel_dim=0),

        ]
    )

    test_ds = prostateMoE(npz_files=test_samples, img_transforms=val_img_transform, seg_transforms=val_seg_transform)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,num_workers=0, pin_memory=True)

    return test_loader

def test(args):
    # load model
    model = _get_model(args)
    model.eval()

    # test loader
    test_loader = _get_loader(args)
    with torch.no_grad():
        post_trans = transforms.Compose([transforms.Activations(sigmoid=True), transforms.AsDiscrete(threshold=0.5)])
        for test_batch in test_loader:
            test_inputs = test_batch["image"].to(args.device)
            name = test_batch['name'][0].split('/')[-1]
            test_outputs = sliding_window_inference(
                inputs=test_inputs,
                roi_size=(128, 128, 16),
                sw_batch_size=1,
                predictor=model,
                overlap=0.5,
                inference=True,
                modality=args.modality
            )
            test_outputs = [post_trans(i) for i in decollate_batch(test_outputs)]
            test_outputs = test_outputs[0][1]
            # save it
            test_outputs = test_outputs.cpu().detach().numpy()
            np.savez(os.path.join(args.save_root, name), pred=test_outputs)
            print(name, test_outputs.shape, np.max(test_outputs))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--modality', default=3, type=int)
    parser.add_argument('--epoch', default=1000, type=int)
    parser.add_argument('--model_name', default=None, type=str)
    parser.add_argument('--file_root', default="/Users/qixinhu/Project/CUHK/Prostate/PAIsData/0426/qixin/SEG", type=str)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    args.device = device

    # save root 
    args.save_root = f"out/MoE_BS1_LR0.0001"
    test(args)