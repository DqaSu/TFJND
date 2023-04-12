from torch.utils import data
from torchvision import transforms
from data_generator import MyDataSet


def make_loader(dataset, batch_size, shuffle=True, num_workers=1, drop_last=False):
    if dataset is None:
        raise AssertionError('Dataset should not be None.')
    else:
        loader = data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
        )
    return loader