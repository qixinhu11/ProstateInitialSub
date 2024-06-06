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
from dataset.prostateSegv2 import prostateSegv2
from models.Unet3D import UNet3D
from monai.networks.nets import SegResNet
import warnings
warnings.filterwarnings("ignore")

def dice_score(preds, labels, spe_sen=False):  # on GPU
    ### preds: w,h,d; label: w,h,d
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    preds = torch.where(preds > 0.5, 1., 0.)
    predict = preds.contiguous().view(1, -1)
    target = labels.contiguous().view(1, -1)

    tp = torch.sum(torch.mul(predict, target))
    fn = torch.sum(torch.mul(predict!=1, target))
    fp = torch.sum(torch.mul(predict, target!=1))
    tn = torch.sum(torch.mul(predict!=1, target!=1))

    den = torch.sum(predict) + torch.sum(target) + 1

    dice = 2 * tp / den
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    specificity = tn/(fp + tn)

    # print(dice, recall, precision)
    if spe_sen:
        return dice, recall, precision, specificity
    else:
        return dice, recall, precision

def _get_model(args):

    if args.model_name == "unet":
        model = UNet3D(in_channel=3, n_class=2)
    else:
        model = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=3,
            out_channels=3,
            dropout_prob=0.2,
        )

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
    with open('./cspca_test.txt', 'r') as f:
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
            transforms.RandFlip(prob=0.2, spatial_axis=0),
            transforms.RandFlip(prob=0.2, spatial_axis=1),
            transforms.RandFlip(prob=0.2, spatial_axis=2),
            transforms.RandScaleIntensity(factors=0.2, prob=0.2),
            transforms.RandShiftIntensity(offsets=0.2, prob=0.2),
        ]
    )
    train_seg_transform = transforms.Compose(
        [
            transforms.EnsureChannelFirst(channel_dim=0),
            transforms.RandSpatialCrop(roi_size=[x, y, z], random_size=False),
            transforms.RandFlip(prob=0.2, spatial_axis=0),
            transforms.RandFlip(prob=0.2, spatial_axis=1),
            transforms.RandFlip(prob=0.2, spatial_axis=2),
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

    train_ds = prostateSegv2(npz_files=train_samples, modality=args.modality, img_transforms=train_img_transform, seg_transforms=train_seg_transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,num_workers=0, pin_memory=True)

    test_ds = prostateSegv2(npz_files=test_samples, modality=args.modality, img_transforms=val_img_transform, seg_transforms=val_seg_transform)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,num_workers=0, pin_memory=True)

    return train_loader, test_loader


def train(args, train_loader, model, optimizer, loss_function):
    model.train()
    loss_avg = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        x, y = batch["image"].to(args.device), batch["label"].float().to(args.device)
        logit_map = model(x)
        loss = loss_function(logit_map, y)
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

def validation(args, test_loader, model):
    model.eval()
    with torch.no_grad():
        post_trans = transforms.Compose([transforms.Activations(sigmoid=True), transforms.AsDiscrete(threshold=0.5)])
        all_dice = 0
        idx = 0
        for val_data in test_loader:
            val_inputs, val_labels = (
                val_data["image"].to(args.device),
                val_data["label"].to(args.device),
            )
            name = val_data['name'][0]
            val_outputs = sliding_window_inference(
                inputs=val_inputs,
                roi_size=(args.x, args.y, args.z),
                sw_batch_size=1,
                predictor=model,
                overlap=0.1,
            )
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]

            val_outputs = val_outputs[0][2]
            val_labels  = (val_labels[0][0] == 2).float()
            dice, _, _ = dice_score(val_outputs, val_labels)

            with open( os.path.join(args.root_dir, 'log.txt'), 'a') as f:
                print(name, f"Dice: {dice.item()}", file=f)
            all_dice += dice.item()
            idx += 1
        metric = all_dice / idx

    return metric

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
            print("=="*20, "Start validation.")
            with open( os.path.join(args.root_dir, 'log.txt'), 'a') as f:
                print("=="*20, "Start validation.", file=f)
            avg_dice = validation(args,test_loader,model)
            writer.add_scalar("validation_avg_dice", avg_dice, args.epoch)
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
    parser.add_argument('--batch_size', default=4, type=int)

    parser.add_argument('--max_epoch', default=2000, type=int)
    parser.add_argument('--eval_every', default=50, type=int)
    parser.add_argument('--warmup_epoch', default=50, type=int)

    # new args
    parser.add_argument('--modality', default=0, type=int)
    parser.add_argument('--model_name', default=None, type=str)

    parser.add_argument('--file_root', default="/Users/qixinhu/Project/CUHK/Prostate/PAIsData/0426/qixin/SEGv2", type=str)
    parser.add_argument('--x', default=128, type=int)
    parser.add_argument('--y', default=128, type=int)
    parser.add_argument('--z', default=8, type=int)


    args = parser.parse_args()
    if args.model_name:
        args.root_dir = 'out/' + f'{args.model_name}_BS{args.batch_size}_LR{args.lr}_MOD{args.modality}'
    else:
        args.root_dir = 'out/' + f'V2_BS{args.batch_size}_LR{args.lr}_MOD{args.modality}'
    args.epoch = 0

    print("MAIN Argument values:")
    for k, v in vars(args).items():
        print(k, '=>', v)
    print('-----------------')

    main(args)
