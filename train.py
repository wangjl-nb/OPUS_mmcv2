import argparse
import importlib
import os
import os.path as osp
import time

import torch
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

import utils


def main():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', required=True)
    parser.add_argument('--override', nargs='+', action=DictAction)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    args = parser.parse_args()

    from mmdet3d.utils import register_all_modules
    register_all_modules(init_default_scope=True)

    cfg = Config.fromfile(args.config)
    if args.override is not None:
        cfg.merge_from_dict(args.override)
    importlib.import_module('models')
    importlib.import_module('loaders')

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = str(args.world_size)

    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    def _scale_dataloader_batch_size(loader_cfg, world_size):
        if loader_cfg is None or world_size <= 1:
            return
        if isinstance(loader_cfg, (list, tuple)):
            for item in loader_cfg:
                _scale_dataloader_batch_size(item, world_size)
            return
        if isinstance(loader_cfg, dict) and 'batch_size' in loader_cfg:
            loader_cfg['batch_size'] = max(1, loader_cfg['batch_size'] // world_size)

    timestamp = time.strftime('%Y-%m-%d/%H-%M-%S', time.localtime(time.time()))

    resume_from = cfg.get('resume_from', None)
    if resume_from:
        if not osp.isfile(resume_from):
            raise FileNotFoundError(resume_from)
        cfg.work_dir = osp.dirname(resume_from)
        cfg.load_from = resume_from
        cfg.resume = True
    else:
        run_name = osp.splitext(osp.split(args.config)[-1])[0]
        run_name = f'{run_name}_{time.strftime("%Y-%m-%d/%H-%M-%S", time.localtime(time.time()))}'
        cfg.work_dir = osp.join('outputs', cfg.model['type'], run_name)
        cfg.resume = False

    cfg.launcher = 'pytorch' if world_size > 1 else 'none'

    if cfg.get('batch_size') is not None and cfg.get('train_dataloader') is not None:
        cfg.train_dataloader.batch_size = max(1, cfg.batch_size // world_size)

    # Treat val/test batch_size as global and scale by world_size.
    _scale_dataloader_batch_size(cfg.get('val_dataloader'), world_size)
    _scale_dataloader_batch_size(cfg.get('test_dataloader'), world_size)

    if 'randomness' not in cfg:
        cfg.randomness = dict(seed=0, deterministic=True)

    runner = Runner.from_cfg(cfg)
    if runner.rank == 0:
        utils.backup_code(cfg.work_dir)
    runner.train()


if __name__ == '__main__':
    main()
