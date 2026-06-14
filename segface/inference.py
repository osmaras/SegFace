"""
SegFace inference wrapper.

Provides SegFaceParser — a simple class for running face segmentation
on arbitrary images using pretrained SegFace models.

CelebAMask-HQ class indices (19 classes):
    0  = background
    1  = neck
    2  = skin
    3  = cloth
    4  = l_ear       5  = r_ear
    6  = l_brow      7  = r_brow
    8  = l_eye       9  = r_eye
    10 = nose
    11 = mouth
    12 = l_lip       13 = u_lip
    14 = hair
    15 = eye_g       16 = hat       17 = ear_r      18 = neck_l
"""

import os
import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F

# Ensure repo root is importable (for network/, loss/, utils/)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from network import get_model

# ImageNet normalization constants
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Backward-compatible alias for callers that reference "segface.inference"
SegFaceParser = None  # defined below


class SegFaceParser:
    """
    Inference wrapper for SegFace face-parsing models.

    Usage::

        parser = SegFaceParser(checkpoint="model_299.pt", device="cuda")
        mask = parser.parse_image(bgr_image)   # (H, W) uint8, class indices
    """

    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda",
        backbone: str = "segface_celeb",
        model_name: str = "swin_base",
        input_resolution: int = 512,
    ):
        self.device = device
        self.resolution = input_resolution
        self.num_classes = 19 if "celeb" in backbone else (11 if "lapa" in backbone or "helen" in backbone else 19)

        # Build model graph
        self.model = get_model(backbone, input_resolution, model_name)

        # Load weights (handles DataParallel "module." prefix)
        state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.to(device).eval()

        # Pre-place normalization buffers on the target device
        self._mean = _IMAGENET_MEAN.to(device)
        self._std = _IMAGENET_STD.to(device)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def parse_image(self, image: np.ndarray) -> np.ndarray:
        """
        Segment a face image.

        Parameters
        ----------
        image : np.ndarray
            BGR image as loaded by ``cv2.imread``, shape (H, W, 3).

        Returns
        -------
        np.ndarray
            Per-pixel class-index map, shape (H, W), dtype ``uint8``.
            See the module docstring for the class-index table.
        """
        orig_h, orig_w = image.shape[:2]

        # BGR → RGB, resize to model resolution, scale to [0, 1]
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0

        # (H, W, 3) → (1, 3, H, W) tensor
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        tensor = (tensor - self._mean) / self._std

        # Forward: labels/dataset args are unused at inference time
        logits = self.model(tensor, None, None)

        # Softmax → argmax at model resolution
        probs = F.interpolate(logits, size=(self.resolution, self.resolution),
                              mode="bilinear", align_corners=False).softmax(dim=1)
        preds = torch.argmax(probs, dim=1)  # (1, H, W)

        # Resize back to original image dimensions
        preds = F.interpolate(preds.unsqueeze(1).float(),
                              size=(orig_h, orig_w), mode="nearest").squeeze(1)

        return preds.cpu().numpy().astype(np.uint8)[0]  # (H, W)
