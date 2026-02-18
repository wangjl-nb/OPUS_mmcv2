from mmengine.dist import is_main_process
from mmengine.hooks import CheckpointHook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class StrictCheckpointHook(CheckpointHook):
    """Checkpoint hook that refuses to overwrite existing checkpoint files."""

    def __init__(self, fail_if_exists: bool = True, **kwargs):
        self.fail_if_exists = fail_if_exists
        super().__init__(**kwargs)

    def _save_checkpoint_with_step(self, runner, step, meta):
        if self.fail_if_exists and is_main_process():
            ckpt_filename = self.filename_tmpl.format(step)
            ckpt_path = self.file_backend.join_path(self.out_dir, ckpt_filename)
            if self.file_backend.isfile(ckpt_path) or self.file_backend.isdir(ckpt_path):
                raise FileExistsError(
                    f'Checkpoint already exists, refusing to overwrite: {ckpt_path}. '
                    'Please use a new work_dir, resume_from, or remove old checkpoints.'
                )

        super()._save_checkpoint_with_step(runner, step, meta)
