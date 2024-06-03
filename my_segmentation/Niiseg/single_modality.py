import os, torch
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from monai.losses import DiceCELoss
from tensorboardX import SummaryWriter
from monai.data import Dataset, DataLoader
from monai.networks.nets import SegResNet
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    RandFlipd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Spacingd,
    ToTensord,
    SpatialPadd,
    RandCropByLabelClassesd,
)
from utils import ConvertLabelBasedOnClasses
from trainer import trainer
import warnings
warnings.filterwarnings("ignore")

def _get_loader(args):
    train_samples = []
    with open(f'./{args.modality}.txt', 'r') as f:
        lines = f.readlines()
        for name in lines:
            name = name.strip()
            sample = {"label": os.path.join(args.file_root, name, f'{name}_{args.modality}_gt.nii.gz'),
                      "image": os.path.join(args.file_root, name, f'{name}_{args.modality}.nii.gz')}
            train_samples.append(sample)

    test_samples = []
    with open('./cspca_test.txt', 'r') as f:
        lines = f.readlines()
        for name in lines:
            name = name.strip()
            sample = {"label": os.path.join(args.file_root, name, f'{name}_{args.modality}_gt.nii.gz'),
                      "image": os.path.join(args.file_root, name, f'{name}_{args.modality}.nii.gz')}
            test_samples.append(sample)

    train_transform = Compose(
        [
            # load single modality
            LoadImaged(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"]),
            ConvertLabelBasedOnClasses(keys="label"),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            SpatialPadd(keys=["image", "label"], mode=["minimum", "constant"], spatial_size=[args.x, args.y, args.z]),
            RandCropByLabelClassesd(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=[args.x, args.y, args.z],
                    num_classes=2,
                    ratios=[0,1],
                    num_samples=args.num_samples,
                ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ToTensord(keys=["image", "label"]),
        ]
    )
    
    test_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"]),
            ConvertLabelBasedOnClasses(keys="label"),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ToTensord(keys=["image", "label"]),
        ]
    )

    train_ds = Dataset(data=train_samples, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    test_ds = Dataset(data=test_samples, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,num_workers=0, pin_memory=True)

    return train_loader, test_loader


def main(args):
    # create writer
    writer = SummaryWriter(log_dir=args.root_dir)
    print('Writing Tensorboard logs to ', args.root_dir)

    # build the model
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=1,
        out_channels=2,
        dropout_prob=0.2,
    )
    model.to(args.device)
    print("The model is now on the CUDA: ", next(model.parameters()).is_cuda)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    torch.backends.cudnn.benchmark = True
    loss_function = DiceCELoss(to_onehot_y=True,sigmoid=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epoch, max_epochs=args.max_epoch)
    train_loader, test_loader = _get_loader(args)

    # start training
    trainer(
        args,
        scheduler,
        writer,
        model,
        optimizer,
        loss_function,
        train_loader,
        test_loader
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--num_samples', default=2, type=int)

    parser.add_argument('--max_epoch', default=2000, type=int)
    parser.add_argument('--eval_every', default=50, type=int)
    parser.add_argument('--warmup_epoch', default=50, type=int)

    # new args
    parser.add_argument('--modality', default="T2W", type=str)

    parser.add_argument('--file_root', default="/Users/qixinhu/Project/CUHK/Prostate/PAIsData/0426/qixin/SEG", type=str)
    parser.add_argument('--x', default=96, type=int)
    parser.add_argument('--y', default=96, type=int)
    parser.add_argument('--z', default=32, type=int)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.root_dir = 'out/' + f'SingleModality_BS{args.batch_size}_LR{args.lr}_{args.modality}'
    args.epoch = 0

    print("MAIN Argument values:")
    for k, v in vars(args).items():
        print(k, '=>', v)
    print('-----------------')
    main(args)