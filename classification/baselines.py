import os, torch, random, monai
import numpy as np
import pandas as pd
from monai import transforms
# from monai.data import DataLoader
from torch.utils.data import DataLoader
from trainer import trainer
from dataset.prostateDataset import prostateDataset
# from models.vit_3d import ViT
from oversampling.imbalanced import ImbalancedDatasetSampler
from optimizers.lr_scheduler import WarmupCosineSchedule,LinearWarmupCosineAnnealingLR


def _get_data_loader(args):
    file_root = args.file_root
    labels_dict = np.load(os.path.join(file_root,"case_level_label.npz"), allow_pickle=True)['label'].item()
    train_samples = []
    with open(os.path.join(file_root,'train_case_level.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            sample = line.strip()
            train_samples.append(sample)

    val_samples = []
    with open(os.path.join(file_root,'test_case_level.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            sample = line.strip()
            val_samples.append(sample)
        
    train_images = []
    train_labels = []
    for sample in train_samples:
        train_images.append(os.path.join(file_root, args.modality, sample + '.npz'))
        train_labels.append(labels_dict[sample])
    train_labels = np.array(train_labels, dtype=float)
    train_labels = torch.FloatTensor(train_labels)

    val_images = []
    val_labels = []
    for sample in val_samples:
        val_images.append(os.path.join(file_root, args.modality, sample + '.npz'))
        val_labels.append(labels_dict[sample])
    val_labels = np.array(val_labels, dtype=float)
    val_labels = torch.FloatTensor(val_labels)  

    train_img_transform = transforms.Compose(
        [
            transforms.EnsureChannelFirst(channel_dim="no_channel"),
            # transforms.Resize(spatial_size=(x, y, z), mode="area"),
            transforms.RandFlip(prob=0.2, spatial_axis=0),
            transforms.RandFlip(prob=0.2, spatial_axis=1),
            transforms.RandFlip(prob=0.2, spatial_axis=2),
            transforms.RandRotate90(prob=0.2, max_k=3),
            transforms.RandScaleIntensity(factors=0.15, prob=0.3),
            transforms.RandShiftIntensity(offsets=0.15, prob=0.3),

            # Add more intensity-based transform
            transforms.RandAdjustContrast(prob=0.2),
            # transforms.RandHistogramShift(prob=0.2),
            # transforms.RandGibbsNoise(prob=0.2),
            # transforms.RandKSpaceSpikeNoise(prob=0.2)
            
        ]
    )
    # add more intensity-based transform.
    train_seg_transform = transforms.Compose(
        [
            transforms.EnsureChannelFirst(channel_dim="no_channel"),
            # transforms.Resize(spatial_size=(x, y, z), mode="nearest"),
            transforms.RandFlip(prob=0.2, spatial_axis=0),
            transforms.RandFlip(prob=0.2, spatial_axis=1),
            transforms.RandFlip(prob=0.2, spatial_axis=2),
            transforms.RandRotate90(prob=0.2, max_k=3),
        ]
    )

    val_img_transform = transforms.Compose(
        [
            transforms.EnsureChannelFirst(channel_dim="no_channel"),
            # transforms.Resize(spatial_size=(x, y, z), mode="area"),
        ]
    )
    val_seg_transform = transforms.Compose(
        [
            transforms.EnsureChannelFirst(channel_dim="no_channel"),
            # transforms.Resize(spatial_size=(x, y, z), mode="nearest"),
        ]
    )

    train_ds = prostateDataset(npz_files=train_images, labels=train_labels, img_transforms=train_img_transform, seg_transforms=None)
    train_sampler = ImbalancedDatasetSampler(train_ds, torch.argmax(train_labels,dim=1))
    train_loader = DataLoader(train_ds, sampler=train_sampler,
                              batch_size=args.batch_size, num_workers=0, pin_memory=True)

    
    val_ds = prostateDataset(npz_files=val_images, labels=val_labels, img_transforms=val_img_transform, seg_transforms=None)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False,num_workers=0, pin_memory=True)

    return train_loader, val_loader

def _get_models(args):
    if args.model_name == "densenet121":
        model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=args.num_cls)
    elif args.model_name == "densenet169":
        model = monai.networks.nets.DenseNet169(spatial_dims=3, in_channels=1, out_channels=args.num_cls)
    elif 'efficientnet' in args.model_name:
        model = monai.networks.nets.EfficientNetBN(model_name=args.model_name, spatial_dims=3, in_channels=1, out_channels=args.num_cls)
    # elif args.model_name == 'vit':
    #     from vit_pytorch.vit_3d import ViT
    #     model = ViT(
    #         image_size = args.resize_x,          # image size
    #         frames = args.resize_z,               # number of frames
    #         channels=1,
    #         image_patch_size = 16,     # image patch size
    #         frame_patch_size = 16,      # frame patch size
    #         num_classes = args.num_cls,
    #         dim = 1024,
    #         depth = 6,
    #         heads = 8,
    #         mlp_dim = 2048,
    #         dropout = 0.1,
    #         emb_dropout = 0.1
    #     )
    elif 'resnet' in args.model_name:
        from models.resnet import generate_model
        n_classes = args.num_cls
        model_depth = int(args.model_name[6:])
        model = generate_model(model_depth=model_depth,
                               input_H=args.H, input_W=args.W, input_D=args.D, n_classes=n_classes)
    else:
        raise RuntimeError("Do not support the baseline method!")
    
    return model

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='medical segmentation contest')

    # training parameters
    parser.add_argument('--max_epochs', default=1000, type=int)
    parser.add_argument('--val_every', default=100, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--model_name', default="resnet18", type=str)
    parser.add_argument('--num_cls', default=2, type=int)
    
    # saving parameters
    parser.add_argument('--file_root', default="/Users/qixinhu/Project/CUHK/Prostate/PAIsData/0426/qixin/SEG", type=str)
    parser.add_argument('--log_dir', default="runs", type=str)
    parser.add_argument('--modality', default="T2W", type=str)

    # augmentation parameters
    parser.add_argument('--H', default=128, type=int)
    parser.add_argument('--W', default=128, type=int)
    parser.add_argument('--D', default=12, type=int)
        

    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    args.device = device

    # Print All Config
    print("MAIN Argument values:")
    for k, v in vars(args).items():
        print(k, '=>', v)
    print('-----------------')

    # loader
    train_loader, val_loader = _get_data_loader(args)
    # # model, optimizer and loss function
    model = _get_models(args)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=10, max_epochs=args.max_epochs
        )
    
    loss_function = torch.nn.BCEWithLogitsLoss()
    # loss_function = torch.nn.CrossEntropyLoss()

    trainer(model, train_loader, val_loader, optimizer, scheduler, loss_function, args)

if __name__ == "__main__":
    setup_seed()
    main()