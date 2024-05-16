import numpy as np 
from monai.transforms import Randomizable, apply_transform
from monai.utils import MAX_SEED
from torch.utils.data import Dataset


class prostateMoE(Dataset, Randomizable):
    def __init__(self, npz_files, img_transforms=None, seg_transforms=None) -> None:
        super().__init__()
        self.npz_files = npz_files
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
        img = data['img'].astype(np.float32)
        mask = data['mask'].astype(np.float32)

        if self.img_transforms is not None:
            if isinstance(self.img_transforms, Randomizable):
                self.img_transforms.set_random_state(seed=self._seed)            
            img = apply_transform(self.img_transforms, img, map_items=False)

        if self.seg_transforms is not None:
            if isinstance(self.seg_transforms, Randomizable):
                self.seg_transforms.set_random_state(seed=self._seed)            
            mask = apply_transform(self.seg_transforms, mask, map_items=False)

        batch_data['name']  = self.npz_files[index]
        batch_data['image'] = img
        batch_data['label'] = mask

        return batch_data