import os, torch, csv
import numpy as np
from monai import transforms
from torch.utils.data import DataLoader
from dataset.prostateDataset import prostateDataset

import warnings
warnings.filterwarnings("ignore")


def _get_data_loader(modality):
    file_root = "../data/preprocess"
    file_list = sorted(os.listdir(os.path.join(file_root, modality)))
            
    test_images = []
    for sample in file_list:
        test_images.append(os.path.join(file_root, modality, sample))
    test_img_transform = transforms.Compose(
        [
            transforms.EnsureChannelFirst(channel_dim="no_channel"),
        ]
    )
    test_ds = prostateDataset(npz_files=test_images, img_transforms=test_img_transform, seg_transforms=None)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,num_workers=0, pin_memory=True)
    return test_loader


def _get_val_loader(args):
    file_root = args.file_root
    labels_dict = np.load(os.path.join(file_root,"case_level_label.npz"), allow_pickle=True)['label'].item()
    val_samples = []
    with open(os.path.join(file_root,'test_case_level.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            sample = line.strip()
            val_samples.append(sample)
        
    val_images = []
    val_labels = []
    for sample in val_samples:
        val_images.append(os.path.join(file_root, args.modality, sample + '.npz'))
        val_labels.append(labels_dict[sample])
    val_labels = np.array(val_labels, dtype=float)
    val_labels = torch.FloatTensor(val_labels)  


    val_img_transform = transforms.Compose(
        [
            transforms.EnsureChannelFirst(channel_dim="no_channel"),
            # transforms.Resize(spatial_size=(x, y, z), mode="area"),
        ]
    )

    
    val_ds = prostateDataset(npz_files=val_images, labels=val_labels, img_transforms=val_img_transform, seg_transforms=None)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,num_workers=0, pin_memory=True)

    return val_loader


def test_one_model(model, test_loader):
    model.eval()
    name_list = []
    y_pred = []
    with torch.no_grad():
        for test_data in test_loader:
            test_name = test_data['name'][0].split('/')[-1].split('.')[0]
            test_images  = test_data['image'].to("cuda")
            test_preds = model(test_images)
            test_preds  = test_preds.detach().cpu().numpy()
            y_pred.append(np.argmax(test_preds))
            name_list.append(test_name)

    return name_list, y_pred


def voting_models():

    from models.resnet import generate_model
    n_classes = 2
    model_depth = 18

    all_preds = {}
    preds = []
    for modality in ["T2W", "DWI", "ADC", "T2W_ADC", "T2W_DWI", "DWI_ADC", "ALL"]:
        val_loader = _get_data_loader(modality)
        if modality in ["T2W", "DWI", "ADC"]:
            input_D = 12
        elif modality in ["T2W_ADC", "T2W_DWI", "DWI_ADC"]:
            input_D = 24
        else:
            input_D = 36
            
        model = generate_model(model_depth=model_depth,
                                input_H=128, input_W=128, input_D=input_D, n_classes=n_classes)
        model.cuda()
        # loading model
        model.load_state_dict(torch.load(f"runs/{modality}/resnet18/model_best.pt"))
        # exam the model
        print("-" * 20)
        print(f"Start to Evaluation {modality}")
        name_list, y_pred = test_one_model(model, val_loader)
        print(f"{modality} prediction:", y_pred)
        all_preds[modality]=y_pred
        preds.append(y_pred)


    preds = np.array(preds)
    preds = np.amax(preds,axis=0).tolist()
    print("Voting prediction:", preds)

    # save the results
    header = ["name", "T2W", "DWI", "ADC", "T2W_ADC", "T2W_DWI", "DWI_ADC", "ALL", "Voting"]
    rows   = []
    for idx in range(len(name_list)):
        row = [name_list[idx]]
        for modality in ["T2W", "DWI", "ADC", "T2W_ADC", "T2W_DWI", "DWI_ADC", "ALL"]:
            row.append(all_preds[modality][idx])
        row.append(preds[idx])
        rows.append(row)

    # save metrics to cvs file
    csv_name = '../classification_prediction.csv'
    with open(csv_name, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


if __name__ == "__main__":
    voting_models()

