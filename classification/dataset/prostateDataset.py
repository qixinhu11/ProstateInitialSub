from torch.utils.data import Dataset
import numpy as np 
from monai.transforms import Randomizable, apply_transform
from monai.utils import MAX_SEED

class prostateDataset(Dataset, Randomizable):
    def __init__(self, npz_files, labels=None, img_transforms=None, seg_transforms=None) -> None:
        super().__init__()
        self.npz_files = npz_files
        self.labels    = labels 
        self.img_transforms = img_transforms
        self.seg_transforms = seg_transforms
    
    def __len__(self) -> int:
        return len(self.npz_files)

    def randomize(self) -> None:
        self._seed = self.R.randint(MAX_SEED, dtype="uint32")

    def __getitem__(self, index: int):
        self.randomize()

        batch_data = {}
        # load npz_file
        data = np.load(self.npz_files[index])
        img = data['img']

        if self.img_transforms is not None:
            if isinstance(self.img_transforms, Randomizable):
                self.img_transforms.set_random_state(seed=self._seed)            
            img = apply_transform(self.img_transforms, img, map_items=False)

        if self.labels is not None:
            label = self.labels[index]
            batch_data["label"] = label

        batch_data['name']  = self.npz_files[index]
        batch_data['image'] = img
        return batch_data