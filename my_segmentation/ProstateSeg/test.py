import os, json, torch, monai
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from monai.losses import DiceCELoss, DiceLoss
from tensorboardX import SummaryWriter
from monai.inferers import sliding_window_inference
from monai.data import DataLoader, decollate_batch
from monai import transforms
from dataset.prostateSeg import prostateSeg
from models.Unet3D import UNet3D
from monai.networks.nets import SegResNet
import warnings
warnings.filterwarnings("ignore")

def _get_model(args):
    # model = UNet3D(in_channel=3)
    if args.modality in [0, 1, 2]:
        in_channels=1
    else:
        in_channels=3

    if args.model_name == "unet":
        model = UNet3D(in_channel=in_channels, n_class=2)
        model_path = f"out/{args.model_name}_BS1_LR0.0001_MOD{args.modality}/model_{args.epoch}.pth"
    else:
        model = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=in_channels,
            out_channels=2,
            dropout_prob=0.2,
        )
        model_path = f"out/BS1_LR0.0001_MOD{args.modality}/model_{args.epoch}.pth"
    model.load_state_dict(torch.load(model_path)['net'])
    model.to(args.device)
    return model
    
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

    test_ds = prostateSeg(npz_files=test_samples, modality=args.modality, img_transforms=val_img_transform, seg_transforms=val_seg_transform)
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
    args.save_root = f"out/BS1_LR0.0001_MOD{args.modality}"
    test(args)