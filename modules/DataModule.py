import h5py
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from utils.utils import bin2dec, dec2bin_numpy


class CenterCrop512(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.size = 512

    def forward(self, img):
        return center_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size=512)'


class BitwiseTransform(torch.nn.Module):
    def __init__(self, single_bit):
        super().__init__()
        self.single_bit = single_bit

    def forward(self, img):
        return dec2bin_numpy(img) if self.single_bit is None else \
            dec2bin_numpy(img)[:, :, self.single_bit]

    def __repr__(self):
        return self.__class__.__name__


def center_crop(img, crop_size=512):
    y_size, x_size = img.shape[-2:]
    x_start = x_size // 2 - (crop_size // 2)
    y_start = y_size // 2 - (crop_size // 2)
    if len(img.shape) == 2:
        return img[y_start:y_start + crop_size, x_start:x_start + crop_size]
    else:
        return img[:, y_start:y_start + crop_size, x_start:x_start + crop_size]


class PUFDataset(Dataset):
    def __init__(self, folder, ids, transform, do_normalize=True):
        self.data_file = f"data/{folder}/data.h5"
        self._h5_gen = None
        self.do_normalize = do_normalize

        with h5py.File(self.data_file, 'r') as data:
            self.min_v = data.get("min")[()]
            self.max_v = data.get("max")[()]

        self.normalize = lambda response: 2 * (response - self.min_v) / (
                self.max_v - self.min_v) - 1
        self.denormalize = lambda response: ((response + 1) * (
                self.max_v - self.min_v) / 2) + self.min_v

        self.folder = folder
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if self._h5_gen is None:
            self._h5_gen = self._get_generator(self.data_file)
            next(self._h5_gen)

        challenge, response = self._h5_gen.send(self.ids[idx])
        if self.transform:
            response = self.transform(response)
        response = torch.tensor(response, dtype=torch.float)
        challenge = torch.tensor(challenge, dtype=torch.float)
        if self.do_normalize:
            response = self.normalize(response)
        return challenge, response

    def _get_generator(self, path):
        with h5py.File(path, 'r') as data:
            index = yield
            while True:
                c = data.get("challenges")[index]
                r = data.get("responses")[index]
                index = yield c, r


class PUFDataModule(LightningDataModule):
    def __init__(self, batch_size, folder, training_ids, test_ids,
                 do_crop=True, bitwise=False, single_bit=None):
        super().__init__()
        self.batch_size = batch_size
        self.folder = folder
        self.train_kwargs = {"batch_size": self.batch_size, "num_workers": 8,
                             "pin_memory": True, "shuffle": True}
        self.val_test_kwargs = {"batch_size": self.batch_size, "num_workers": 8,
                                "pin_memory": True, "shuffle": True}
        self.training_ids = training_ids
        self.test_ids = test_ids
        self.do_crop = do_crop
        self.bitwise = bitwise
        self.single_bit = single_bit

    def setup(self):
        ts = []
        if self.do_crop:
            ts.append(CenterCrop512())
        if self.bitwise:
            ts.append(BitwiseTransform(self.single_bit))
        transform = transforms.Compose(ts)

        self.train_dataset = PUFDataset(
            self.folder, self.training_ids, transform,
            do_normalize=not self.bitwise
        )
        self.test_dataset = PUFDataset(
            self.folder, self.test_ids, transform, do_normalize=not self.bitwise
        )

        if self.bitwise:
            self.denormalize = (lambda x: x) if self.single_bit \
                else (lambda x: bin2dec(x))
        else:
            self.denormalize = self.train_dataset.denormalize

    def setup_random_split(self):
        ts = []
        if self.do_crop:
            ts.append(CenterCrop512())
        if self.bitwise:
            ts.append(BitwiseTransform(self.single_bit))
        transform = transforms.Compose(ts)
        dataset = PUFDataset(
            self.folder, self.training_ids + self.test_ids, transform,
            do_normalize=not self.bitwise
        )

        train_length = int(0.9 * len(dataset))
        test_length = len(dataset) - train_length
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(
            dataset, [train_length, test_length]
        )

        self.denormalize = lambda x: bin2dec(x) \
            if self.bitwise else self.train_dataset.denormalize

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.train_kwargs)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, **self.val_test_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.val_test_kwargs)


class PUFDataModuleVarSets(LightningDataModule):
    def __init__(self, batch_size, folder, test_folder, training_ids, test_ids):
        super().__init__()
        self.batch_size = batch_size
        self.folder = folder
        self.test_folder = test_folder
        self.train_kwargs = {"batch_size": self.batch_size, "num_workers": 8,
                             "pin_memory": True, "shuffle": True}
        self.val_test_kwargs = {"batch_size": self.batch_size, "num_workers": 8,
                                "pin_memory": True, "shuffle": True}
        self.training_ids = training_ids
        self.test_ids = test_ids

    def setup(self):
        transform = transforms.Compose([
            CenterCrop512()
        ])
        self.train_dataset = PUFDataset(
            self.folder, self.training_ids, transform
        )
        self.test_dataset = PUFDataset(
            self.test_folder, self.test_ids, transform
        )

        self.denormalize = self.train_dataset.denormalize

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.train_kwargs)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, **self.val_test_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.val_test_kwargs)
