from torch.utils.data import DataLoader, SubsetRandomSampler
from data_load.data_preprocess import IEMOCAPDataset, MELDDataset
import numpy as np
from args_setting import get_args

args = get_args()

def _init_fn(worker_id):
        np.random.seed(int(args.seed)+worker_id)

def get_train_valid_sampler(trainset, valid=0.1, dataset='IEMOCAP'):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_MELD_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False, epoch_ratio=-1):
    trainset = MELDDataset('data/MELD_features_raw1.pkl', epoch_ratio=epoch_ratio)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid, 'MELD')

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset('data/MELD_features_raw1.pkl', train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False, epoch_ratio=-1):
    trainset = IEMOCAPDataset(epoch_ratio=epoch_ratio)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory, worker_init_fn=_init_fn)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory, worker_init_fn=_init_fn)

    return train_loader, valid_loader, test_loader

def load_data(args, batch_size, epoch_ratio=-1):
    """加载数据集"""
    if args.Dataset == 'MELD':
        return get_MELD_loaders(valid=0.0, batch_size=batch_size, num_workers=2, epoch_ratio=epoch_ratio)
    elif args.Dataset == 'IEMOCAP':
        return get_IEMOCAP_loaders(valid=0.0, batch_size=batch_size, num_workers=2, epoch_ratio=epoch_ratio)
    else:
        raise ValueError("Unsupported dataset")
