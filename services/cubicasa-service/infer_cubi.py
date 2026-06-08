"""Standalone CubiCasa5k inference on an arbitrary floor-plan image (CPU).
Usage: python infer_cubi.py <image> <out_rooms.png> <out_icons.png>
Outputs colorized room-type segmentation and icon (door/window/fixture) segmentation."""
import sys
import numpy as np
import torch
import torch.nn.functional as F
from skimage import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# scipy>=1.9 made stats.mode return a scalar .mode; CubiCasa's post-processing expects
# the old array form (.mode[0]). Restore it with keepdims.
import scipy.stats as _ss
_orig_mode = _ss.mode
def _mode_compat(a, axis=0, **kw):
    kw.pop("keepdims", None)
    return _orig_mode(a, axis=axis, keepdims=True)
_ss.mode = _mode_compat

from floortrans.models import get_model
from floortrans.loaders import RotateNTurns
from floortrans.plotting import polygons_to_image, discrete_cmap
from floortrans.post_prosessing import split_prediction, get_polygons

try:
    discrete_cmap()
except ValueError:
    pass  # colormaps already registered on import
img_path, out_rooms, out_icons = sys.argv[1], sys.argv[2], sys.argv[3]
n_classes, split, n_rooms, n_icons = 44, [21, 12, 11], 12, 11

model = get_model("hg_furukawa_original", 51)
model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)
ckpt = torch.load("/app/model/model_best_val_loss_var.pkl", map_location="cpu")
model.load_state_dict(ckpt["model_state"])
model.eval()

img = io.imread(img_path)
if img.ndim == 2:
    img = np.stack([img] * 3, -1)
img = img[:, :, :3].astype(np.float32)
img = 2 * (img / 255.0) - 1
H, W = img.shape[:2]
pad_h, pad_w = (32 - H % 32) % 32, (32 - W % 32) % 32
t = torch.tensor(np.moveaxis(img, -1, 0))[None]
t = F.pad(t, (0, pad_w, 0, pad_h))
h2, w2 = t.shape[2], t.shape[3]

rot = RotateNTurns()
rotations = [(0, 0), (1, -1), (2, 2), (-1, 1)]
with torch.no_grad():
    pred = torch.zeros([len(rotations), n_classes, h2, w2])
    for i, (f, b) in enumerate(rotations):
        ri = rot(t, "tensor", f)
        p = model(ri)
        p = rot(p, "tensor", b)
        p = rot(p, "points", b)
        p = F.interpolate(p, size=(h2, w2), mode="bilinear", align_corners=True)
        pred[i] = p[0]
    prediction = torch.mean(pred, 0, True)[:, :, :H, :W]

heatmaps, rooms, icons = split_prediction(prediction, (H, W), split)
polygons, types, room_polygons, room_types = get_polygons((heatmaps, rooms, icons), 0.2, [1, 2])
pol_room_seg, pol_icon_seg = polygons_to_image(polygons, types, room_polygons, room_types, H, W)

for seg, out, nc, cmap in [(pol_room_seg, out_rooms, n_rooms, "rooms"),
                           (pol_icon_seg, out_icons, n_icons, "icons")]:
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(seg, cmap=cmap, vmin=0, vmax=nc - 0.1)
    plt.tight_layout(pad=0)
    plt.savefig(out, bbox_inches="tight", pad_inches=0, dpi=110)
    plt.close()

print("rooms detected:", sorted(set(int(x) for x in np.unique(pol_room_seg))))
print("icons detected:", sorted(set(int(x) for x in np.unique(pol_icon_seg))))
print("done", pol_room_seg.shape)
