from functools import partial
from mmengine.dataset import default_collate, worker_init_fn, DefaultSampler
from mmengine.dist import get_dist_info
from torch.utils.data import DataLoader


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     **kwargs):

    rank, world_size = get_dist_info()
    if dist:
        sampler = DefaultSampler(dataset, shuffle=shuffle, seed=seed)
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=default_collate,
        pin_memory=False,
        worker_init_fn=init_fn,
        **kwargs)

    return data_loader
