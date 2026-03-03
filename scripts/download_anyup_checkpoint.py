#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch


CHECKPOINT_URLS = {
    'anyup': 'https://github.com/wimmerth/anyup/releases/download/checkpoint/anyup_paper.pth',
    'anyup_multi_backbone': 'https://github.com/wimmerth/anyup/releases/download/checkpoint_v2/anyup_multi_backbone.pth',
}

CHECKPOINT_NAMES = {
    'anyup': 'anyup_paper.pth',
    'anyup_multi_backbone': 'anyup_multi_backbone.pth',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Download AnyUp checkpoint to local project path.')
    parser.add_argument(
        '--variant',
        choices=sorted(CHECKPOINT_URLS.keys()),
        default='anyup_multi_backbone',
        help='AnyUp model variant.')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/root/wjl/OPUS_mmcv2/third_party/anyup/checkpoints',
        help='Directory to save checkpoint.')
    parser.add_argument(
        '--force',
        action='store_true',
        help='Redownload even if file already exists.')
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = CHECKPOINT_NAMES[args.variant]
    out_path = output_dir / filename
    if out_path.exists() and not args.force:
        print(f'[Info] checkpoint already exists: {out_path}')
        return

    url = CHECKPOINT_URLS[args.variant]
    print(f'[Info] downloading {args.variant} from {url}')
    torch.hub.download_url_to_file(url, str(out_path), progress=True)
    print(f'[Info] saved checkpoint to {out_path}')


if __name__ == '__main__':
    main()
