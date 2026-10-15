"""MNAD black-box detector wrapper. Copied from 02_MNAD.ipynb -- do not
modify the underlying MNAD architecture; this class only wraps inference."""

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable

logger = logging.getLogger("SmartHomeSurveillance.MNADDetector")


class MNADDetector:
    """Black-box wrapper around a trained MNAD checkpoint.
    Contract: Frame Sequence -> Anomaly Score. Nothing else.

    NOTE: requires MNAD's own modules to already be on sys.path (the caller
    notebook must do `sys.path.insert(0, str(PATHS.mnad_repo))` before
    importing this module, exactly as 02_MNAD.ipynb does).
    """

    def __init__(
        self,
        model_path: Path,
        m_items_path: Path,
        method: str = "pred",
        height: int = 256,
        width: int = 256,
        t_length: int = 5,
        alpha: float = 0.6,
        th: float = 0.01,
        device: torch.device = None,
    ):
        from model.utils import DataLoader as MNADFrameDataLoader  # noqa: F401
        from utils import psnr, anomaly_score_list, anomaly_score_list_inv, score_sum, AUC, point_score

        self._psnr = psnr
        self._anomaly_score_list = anomaly_score_list
        self._anomaly_score_list_inv = anomaly_score_list_inv
        self._score_sum = score_sum
        self._point_score = point_score
        self._MNADFrameDataLoader = MNADFrameDataLoader
        self._transforms = transforms

        assert method in ("pred", "recon"), "method must be 'pred' or 'recon'"
        self.method = method
        self.height = height
        self.width = width
        self.t_length = t_length
        self.time_step = t_length - 1
        self.alpha = alpha
        self.th = th
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading MNAD model from {model_path}")
        self.model = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Loading memory items from {m_items_path}")
        self.m_items = torch.load(m_items_path, map_location=self.device, weights_only=False)

        self.loss_func_mse = nn.MSELoss(reduction="none")

    def score_frames_folder(self, frames_root: Path) -> Dict[str, np.ndarray]:
        """Run MNAD over every clip subfolder inside frames_root."""
        frames_root = Path(frames_root)
        dataset = self._MNADFrameDataLoader(
            str(frames_root),
            self._transforms.Compose([self._transforms.ToTensor()]),
            resize_height=self.height, resize_width=self.width,
            time_step=self.time_step,
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

        clip_names = sorted(p.name for p in frames_root.iterdir() if p.is_dir())
        clip_lengths = {name: len(list((frames_root / name).glob("*.jpg"))) for name in clip_names}

        psnr_list = {name: [] for name in clip_names}
        feature_distance_list = {name: [] for name in clip_names}

        m_items_test = self.m_items.clone()
        label_length = clip_lengths[clip_names[0]]
        video_num = 0

        with torch.no_grad():
            for k, imgs in enumerate(loader):
                if self.method == "pred":
                    if k == label_length - 4 * (video_num + 1):
                        video_num += 1
                        label_length += clip_lengths[clip_names[video_num]]
                else:
                    if k == label_length:
                        video_num += 1
                        label_length += clip_lengths[clip_names[video_num]]

                imgs = Variable(imgs).to(self.device)

                if self.method == "pred":
                    (outputs, feas, updated_feas, m_items_test, softmax_score_query,
                     softmax_score_memory, _, _, _, compactness_loss) = self.model.forward(
                        imgs[:, 0:3 * 4], m_items_test, False
                    )
                    mse_imgs = torch.mean(
                        self.loss_func_mse((outputs[0] + 1) / 2, (imgs[0, 3 * 4:] + 1) / 2)
                    ).item()
                    mse_feas = compactness_loss.item()
                    point_sc = self._point_score(outputs, imgs[:, 3 * 4:])
                else:
                    (outputs, feas, updated_feas, m_items_test, softmax_score_query,
                     softmax_score_memory, compactness_loss) = self.model.forward(
                        imgs, m_items_test, False
                    )
                    mse_imgs = torch.mean(
                        self.loss_func_mse((outputs[0] + 1) / 2, (imgs[0] + 1) / 2)
                    ).item()
                    mse_feas = compactness_loss.item()
                    point_sc = self._point_score(outputs, imgs)

                if point_sc < self.th:
                    query = F.normalize(feas, dim=1).permute(0, 2, 3, 1)
                    m_items_test = self.model.memory.update(query, m_items_test, False)

                psnr_list[clip_names[video_num]].append(self._psnr(mse_imgs))
                feature_distance_list[clip_names[video_num]].append(mse_feas)

        scores_by_clip: Dict[str, np.ndarray] = {}
        for name in clip_names:
            combined = self._score_sum(
                self._anomaly_score_list(psnr_list[name]),
                self._anomaly_score_list_inv(feature_distance_list[name]),
                self.alpha,
            )
            scores_by_clip[name] = np.asarray(combined)
        return scores_by_clip
