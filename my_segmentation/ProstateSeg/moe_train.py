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
from dataset.prostateMoE import prostateMoE
from models.Unet3D import UNet3D
from monai.networks.nets import SegResNet
import warnings
warnings.filterwarnings("ignore")


def _get_model(args):
    from models.MoESeg import MoESeg
    model = MoESeg()
    return model

def _get_loader(args):
    file_root = args.file_root
    train_samples = []
    with open(os.path.join(file_root,'train_case_level.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            sample = os.path.join(file_root, line.strip() + '.npz')
            train_samples.append(sample)

    test_samples = []
    with open(os.path.join(file_root,'test_case_level.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            sample = os.path.join(file_root, line.strip() + '.npz')
            test_samples.append(sample)

    x = args.x
    y = args.y
    z = args.z
    train_img_transform = transforms.Compose(
        [
            transforms.EnsureChannelFirst(channel_dim=0),
            transforms.RandSpatialCrop(roi_size=[x, y, z], random_size=False),
            transforms.RandFlip(prob=0.5, spatial_axis=0),
            transforms.RandFlip(prob=0.5, spatial_axis=1),
            transforms.RandFlip(prob=0.5, spatial_axis=2),
            transforms.RandScaleIntensity(factors=0.1, prob=0.3),
            transforms.RandShiftIntensity(offsets=0.1, prob=0.3),
        ]
    )
    train_seg_transform = transforms.Compose(
        [
            transforms.EnsureChannelFirst(channel_dim=0),
            transforms.RandSpatialCrop(roi_size=[x, y, z], random_size=False),
            transforms.RandFlip(prob=0.5, spatial_axis=0),
            transforms.RandFlip(prob=0.5, spatial_axis=1),
            transforms.RandFlip(prob=0.5, spatial_axis=2),
        ]
    )

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

    train_ds = prostateMoE(npz_files=train_samples, img_transforms=train_img_transform, seg_transforms=train_seg_transform)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,num_workers=0, pin_memory=True)

    test_ds = prostateMoE(npz_files=test_samples, img_transforms=val_img_transform, seg_transforms=val_seg_transform)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,num_workers=0, pin_memory=True)

    return train_loader, test_loader


def train(args, train_loader, model, optimizer, loss_function):
    model.train()
    loss_avg = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        x, y = batch["image"].to(args.device), batch["label"].float().to(args.device)  # [B,3,128,128,16]; [B,3,128,128,16]; [B]
        t2w_y = y[:,0,...].unsqueeze(1)
        dwi_y = y[:,1,...].unsqueeze(1)
        adc_y = y[:,2,...].unsqueeze(1)
        t2w_out, dwi_out, adc_out, moe_out = model(x)
        # loss
        t2w_loss = loss_function(t2w_out, t2w_y)
        dwi_loss = loss_function(dwi_out, dwi_y)
        adc_loss = loss_function(adc_out, adc_y)
        moe_loss = loss_function(moe_out, t2w_y)
        loss = 0.25 * (t2w_loss + dwi_loss + adc_loss + moe_loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Epoch=%d: Training (%d / %d Steps) (DiceCE_loss=%2.5f)" % (
                args.epoch, step, len(train_loader), loss.item())
        )
        loss_avg += loss.item()
        torch.cuda.empty_cache()

    print('Epoch=%d: AVG_DiceCE_loss=%2.5f' % (args.epoch, loss_avg/len(epoch_iterator)))
    return loss_avg/len(epoch_iterator)

def main(args):
    # create writer
    root_dir = args.root_dir
    writer = SummaryWriter(log_dir=root_dir)
    print('Writing Tensorboard logs to ', root_dir)

    # build the model
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    model = _get_model(args)
    model.to(device)
    print("The model is now on the CUDA: ", next(model.parameters()).is_cuda)
    torch.backends.cudnn.benchmark = True
    loss_function = DiceCELoss(to_onehot_y=True,sigmoid=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epoch, max_epochs=args.max_epoch)

    train_loader, test_loader = _get_loader(args)

    while args.epoch < args.max_epoch:
        scheduler.step()
        loss_dicece = train(args,train_loader=train_loader, model=model, optimizer=optimizer, loss_function=loss_function)
        writer.add_scalar("train_dicece_loss", loss_dicece, args.epoch)
        writer.add_scalar('lr', scheduler.get_lr(), args.epoch)

        checkpoint = {
                    "net": model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    "epoch": args.epoch
                }
        torch.save(checkpoint, os.path.join(root_dir, f"last_model.pth"))
        
        if (args.epoch % args.eval_every == 0 and args.epoch != 0): # start to validation
            # print("=="*20, "Start validation.")
            # with open( os.path.join(args.root_dir, 'log.txt'), 'a') as f:
            #     print("=="*20, "Start validation.", file=f)
            # avg_dice = validation(args,test_loader,model)
            # writer.add_scalar("validation_avg_dice", avg_dice, args.epoch)
            checkpoint = {
                    "net": model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    "epoch": args.epoch
                }
            torch.save(checkpoint, os.path.join(root_dir, f"model_{args.epoch}.pth"))
        args.epoch += 1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_samples', default=4, type=int)

    parser.add_argument('--max_epoch', default=2000, type=int)
    parser.add_argument('--eval_every', default=50, type=int)
    parser.add_argument('--warmup_epoch', default=50, type=int)

    # new args
    parser.add_argument('--model_name', default=None, type=str)

    parser.add_argument('--file_root', default="/Users/qixinhu/Project/CUHK/Prostate/PAIsData/0426/qixin/SEG", type=str)
    parser.add_argument('--x', default=128, type=int)
    parser.add_argument('--y', default=128, type=int)
    parser.add_argument('--z', default=16, type=int)


    args = parser.parse_args()

    args.root_dir = 'out/' + f'MoE_BS{args.batch_size}_LR{args.lr}'
    args.epoch = 0

    print("MAIN Argument values:")
    for k, v in vars(args).items():
        print(k, '=>', v)
    print('-----------------')

    main(args)
