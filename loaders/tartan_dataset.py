"""Skeleton dataset for custom TartanGround-style data.

Fill in the methods to load your own annotations, pack them into the
pipeline-friendly dict, and implement evaluation if needed.
"""

from mmengine.dataset import BaseDataset
from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class TartanDataset(BaseDataset):
    """Template dataset.

    TODO:
    - Define __init__ signature you need (e.g., add custom args), and call
      super().__init__(filter_empty_gt=False, *args, **kwargs).
    - Decide which fields your info list contains.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: load annotations now or keep lazy loading.
        # self.data_infos = self.load_annotations(self.ann_file)

    def load_annotations(self, ann_file):
        """Read your annotation file and return data_infos list.

        Args:
            ann_file (str): Path to your preprocessed info file.

        Returns:
            list[dict]: Each dict holds per-frame info used by get_data_info.
        """
        # TODO: implement loading logic (e.g., mmcv.load for pkl/json).
        raise NotImplementedError

    def get_data_info(self, index):
        """Pack one sample for the pipeline.

        The returned dict keys should match the transforms you use
        (e.g., img_filename, img_timestamp, ego2img, pts_filename, etc.).
        """
        # TODO: map self.data_infos[index] into a pipeline-friendly dict.
        raise NotImplementedError

    def evaluate(self, results, runner=None, show_dir=None, **eval_kwargs):
        """Optional: implement metrics.

        If you only need training, you can return an empty dict for now.
        """
        # TODO: compute and return metrics (mIoU/RayIoU) or empty dict.
        return {}
