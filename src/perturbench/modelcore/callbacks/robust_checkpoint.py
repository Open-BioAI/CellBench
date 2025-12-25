"""
Robust ModelCheckpoint callback that handles disk quota errors gracefully.
"""
import logging
from typing import Any
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

log = logging.getLogger(__name__)


class RobustModelCheckpoint(ModelCheckpoint):
    """
    A ModelCheckpoint callback that gracefully handles disk quota errors.
    
    When disk quota is exceeded, this callback will log a warning and skip
    checkpoint saving, allowing training to continue.
    """
    
    def _save_checkpoint(self, trainer: L.Trainer, filepath: str) -> None:
        """
        Override the checkpoint saving method to handle disk quota errors.
        """
        try:
            # Call the parent method to save the checkpoint
            super()._save_checkpoint(trainer, filepath)
        except OSError as e:
            # Check if it's a disk quota error (errno 122)
            if e.errno == 122:  # Disk quota exceeded
                log.warning(
                    f"Disk quota exceeded while saving checkpoint to {filepath}. "
                    f"Skipping checkpoint save. Training will continue. "
                    f"Error: {e}"
                )
                # Don't re-raise the exception, allow training to continue
            else:
                # For other OSErrors, re-raise them
                log.error(f"OSError while saving checkpoint: {e}")
                raise
        except Exception as e:
            # For any other exceptions, log and re-raise
            log.error(f"Unexpected error while saving checkpoint: {e}")
            raise
    
    def _update_best_and_save(
        self,
        current: Any,
        trainer: L.Trainer,
        monitor_candidates: dict[str, Any],
    ) -> None:
        """
        Override to handle disk quota errors during best model saving.
        """
        try:
            super()._update_best_and_save(current, trainer, monitor_candidates)
        except OSError as e:
            if e.errno == 122:  # Disk quota exceeded
                log.warning(
                    f"Disk quota exceeded while saving best checkpoint. "
                    f"Skipping checkpoint save. Training will continue. "
                    f"Error: {e}"
                )
            else:
                log.error(f"OSError while saving best checkpoint: {e}")
                raise
        except Exception as e:
            log.error(f"Unexpected error while saving best checkpoint: {e}")
            raise
    
    def _save_topk_checkpoint(
        self,
        trainer: L.Trainer,
        monitor_candidates: dict[str, Any],
    ) -> None:
        """
        Override to handle disk quota errors during top-k checkpoint saving.
        """
        try:
            super()._save_topk_checkpoint(trainer, monitor_candidates)
        except OSError as e:
            if e.errno == 122:  # Disk quota exceeded
                log.warning(
                    f"Disk quota exceeded while saving top-k checkpoint. "
                    f"Skipping checkpoint save. Training will continue. "
                    f"Error: {e}"
                )
            else:
                log.error(f"OSError while saving top-k checkpoint: {e}")
                raise
        except Exception as e:
            log.error(f"Unexpected error while saving top-k checkpoint: {e}")
            raise

