# New Dataset Templates

This folder collects templates for adding a new Occ3D-style dataset to this project.

## Files

- `my_occ3d_dataset.py`
  - Minimal dataset class template.
  - Implements `load_data_list()` and `get_data_info()` with the keys expected by current pipelines.
- `config_occ3d_template.py`
  - Config template using `dataset_cfg`.
  - Includes examples for dataloader, pipeline, and evaluator wiring.
- `info_schema.md`
  - Detailed key-by-key schema for one `info` sample in `ann_file` (`*.pkl`).
  - Also documents GT `labels.npz` key requirements.
- `validate_ann_info.py`
  - Quick checker for one `ann_file` to validate required keys and shapes.

## Quick Start

1. Copy `my_occ3d_dataset.py` to `loaders/my_occ3d_dataset.py` and rename the class.
2. Copy `config_occ3d_template.py` to your config path and fill in paths/classes/ranges.
3. Ensure your `ann_file` follows `info_schema.md`.
4. Run:

```bash
python templates/new_dataset/validate_ann_info.py --ann-file /path/to/train.pkl
```

5. Train/val smoke test with your new config.

## Notes

- If your metadata fields are already compatible with existing dataset classes in `loaders/`, you may only need config changes.
- If metadata format differs (pose/sweeps/camera struct), you need a dataset adapter class.
