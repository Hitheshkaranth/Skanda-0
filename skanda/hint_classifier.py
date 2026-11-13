"""Hint classifier: a separate chest X-ray classifier whose probabilities write the classifier hints (skanda/hints.py).

  p(abnormal) + the 13 SCORE_CLASSES from one film, at 512-768 px
Why: run 3's score head (a small head on the language model's state at report resolution) ranks the weak classes
little better than chance on the 537 hint-calibration films (AUC: atelectasis 0.61, fibrosis 0.64, emphysema 0.70,
pneumothorax 0.74), so they get no hint or only "possible", and the report rarely names them. Classifiers trained on
NIH ChestX-ray14 reach 0.80-0.93 on these classes (CheXNet). This one trains on every leakage-free NIH film plus
ChestX-Det train (skanda/data/build_nihcls.py) with a masked BCE (-1 = unknown).
Backbones, both licence-clean (no ImageNet weights):
  hiera     SAM 2.1's Hiera-L image encoder, initialised from a Skanda checkpoint's skanda_heads.pt (Apache-2.0,
            Sa2VA), fine-tuned; avg + max pooled stride-32 features → linear
  densenet  DenseNet-121 from scratch (CheXNet's architecture)
Used at read time when a checkpoint has hint_classifier.pt (SkandaModel.load attaches it as model.hint_cls): the hint
comes from these probabilities; the abnormal gate still uses the score head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import SAM_MEAN, SAM_STD, SCORE_CLASSES

N_OUT = 1 + len(SCORE_CLASSES)


def hiera_l():
    from .sam2 import Hiera
    return Hiera(embed_dim=144, num_heads=2, stages=[2, 6, 36, 4], global_att_blocks=[23, 33, 43],
                 window_pos_embed_bkg_spatial_size=[7, 7], window_spec=[8, 4, 16, 8])


class HintClassifier(nn.Module):
    def __init__(self, backbone="hiera", size=512):
        super().__init__()
        self.backbone, self.size = backbone, size
        if backbone == "hiera":
            self.net = hiera_l()
            dim = self.net.channel_list[0]   # deepest stage first
        elif backbone == "densenet":
            import torchvision
            self.net = torchvision.models.densenet121(weights=None).features
            dim = 1024
        else:
            raise ValueError(backbone)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(2 * dim, N_OUT)
        self.register_buffer("mean", torch.tensor(SAM_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(SAM_STD).view(1, 3, 1, 1), persistent=False)

    def init_from_skanda(self, heads_path):
        """Hiera weights from a Skanda checkpoint's SAM 2.1 image encoder."""
        sd = torch.load(heads_path, map_location="cpu")["sam"]
        pre = "sam2_model.image_encoder.trunk."
        trunk = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
        missing, unexpected = self.net.load_state_dict(trunk, strict=False)
        assert not missing and not unexpected, (missing[:5], unexpected[:5])
        return len(trunk)

    def forward(self, x):
        """x: [B, 1 or 3, H, W] in [0, 1] → logits [B, 1 + C]."""
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        x = (x - self.mean) / self.std
        f = self.net(x)[-1] if self.backbone == "hiera" else F.relu(self.net(x))
        z = torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1)
        return self.fc(self.drop(z.float()))

    @torch.no_grad()
    def predict(self, images):
        """PIL images → probabilities [N, 1 + C] (abnormal first), at the training size."""
        import numpy as np
        dev = self.fc.weight.device
        xs = [torch.from_numpy(np.asarray(im.convert("L").resize((self.size, self.size)), dtype=np.float32) / 255.0)
              for im in images]
        x = torch.stack(xs)[:, None].to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            return torch.sigmoid(self(x)).float().cpu()

    def save(self, path, meta=None):
        torch.save({"backbone": self.backbone, "size": self.size, "state": self.state_dict(), "meta": meta or {}}, path)

    @classmethod
    def load(cls, path, map_location="cpu"):
        ck = torch.load(path, map_location=map_location)
        m = cls(ck["backbone"], ck["size"])
        m.load_state_dict(ck["state"])
        m.meta = ck.get("meta", {})
        return m.eval()
