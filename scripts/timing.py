import os
import sys
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, path)

import time
import logging
import argparse
import importlib
import torch
import torch.distributed
import torch.backends.cudnn as cudnn
from mmengine.config import Config, DictAction
from mmengine.dataset import DefaultSampler, pseudo_collate
from mmengine.model.wrappers import MMDataParallel
from mmengine.runner import load_checkpoint, set_random_seed
from mmdet3d.registry import DATASETS, MODELS
from torch.utils.data import DataLoader


def init_logging(filename=None, debug=False):
    logging.root = logging.RootLogger('DEBUG' if debug else 'INFO')
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] - %(message)s')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)


def main():
    parser = argparse.ArgumentParser(description='Validate a detector')
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--num_warmup', default=10)
    parser.add_argument('--samples', default=500)
    parser.add_argument('--log-interval', default=50, help='interval of logging')
    parser.add_argument('--override', nargs='+', action=DictAction)
    args = parser.parse_args()

    # parse configs
    cfgs = Config.fromfile(args.config)
    if args.override is not None:
        cfgs.merge_from_dict(args.override)

    # register custom module
    from mmdet3d.utils import register_all_modules
    register_all_modules(init_default_scope=True)
    importlib.import_module('models')
    importlib.import_module('loaders')

    init_logging(None, cfgs.debug)

    # you need GPUs
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    logging.info('Using GPU: %s' % torch.cuda.get_device_name(0))
    torch.cuda.set_device(0)

    logging.info('Setting random seed: 0')
    set_random_seed(0, deterministic=True)
    cudnn.benchmark = True

    logging.info('Loading validation set from %s' % cfgs.val_dataloader.dataset.data_root)
    val_dataset = DATASETS.build(cfgs.val_dataloader.dataset)
    val_sampler = DefaultSampler(val_dataset, shuffle=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfgs.val_dataloader.batch_size,
        num_workers=cfgs.val_dataloader.num_workers,
        sampler=val_sampler,
        collate_fn=pseudo_collate,
        persistent_workers=cfgs.val_dataloader.get('persistent_workers', False),
    )

    logging.info('Creating model: %s' % cfgs.model.type)
    model = MODELS.build(cfgs.model)
    model.cuda()

    assert torch.cuda.device_count() == 1
    model = MMDataParallel(model, device_ids=[0])

    logging.info('Loading checkpoint from %s' % args.weights)
    load_checkpoint(
        model, args.weights, map_location='cuda', strict=False,
        logger=logging.Logger(__name__, logging.ERROR)
    )
    model.eval()

    pure_inf_time = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            inputs = data['inputs']
            data_samples = data['data_samples']
            model(inputs=inputs, data_samples=data_samples, mode='predict')

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            if i >= args.num_warmup:
                pure_inf_time += elapsed
                if (i + 1) % args.log_interval == 0:
                    fps = (i + 1 - args.num_warmup) / pure_inf_time
                    print(f'Done sample [{i + 1:<3}/ {args.samples}], '
                        f'fps: {fps:.1f} sample / s')

            if (i + 1) == args.samples:
                break


if __name__ == '__main__':
    main()
