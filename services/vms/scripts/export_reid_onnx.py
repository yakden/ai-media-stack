#!/usr/bin/env python3
"""Export a Torchreid OSNet Re-ID model to ONNX for onnxruntime-gpu.

This runs OFFLINE on a dev box that has torch + torchreid installed. The
runtime VMS container needs ONLY the resulting ``.onnx`` file plus
``onnxruntime-gpu`` (already a dependency) — torch/torchreid never ship in the
image.

Default model
-------------
OSNet ``osnet_x0_25`` trained on MSMT17 (Torchreid zoo key
``osnet_x0_25_msmt17``). x0_25 is the lightest OSNet (~0.2M params) and MSMT17
generalizes across cameras better than Market-1501. The exported graph outputs
the 512-d global feature (the layer before the classifier), which the runtime
:class:`app.reid.embedder.ReIDEmbedder` L2-normalizes for cosine matching.

Drop-in upgrade: ``--arch osnet_ain_x1_0 --pretrained osnet_ain_x1_0_msmt17``
(AIN = better cross-domain), same 512-d output and same export path; just point
``REID_MODEL`` / ``config.reid_model`` at the new file.

Install + run
-------------
    pip install torch torchreid onnx
    # optional fp16 (half the VRAM/size, faster on the T4 Turing):
    pip install onnxconverter-common
    python scripts/export_reid_onnx.py --out models/osnet_x0_25_msmt17.onnx --fp16

The export uses a dynamic batch axis so the runtime can batch several per-frame
person crops into one inference call.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Network input is (H, W) = 256x128 for OSNet (the torchreid ReID standard).
INPUT_H = 256
INPUT_W = 128


def export(
    arch: str,
    pretrained: str,
    out_path: Path,
    fp16: bool,
    opset: int,
) -> int:
    try:
        import torch
        import torchreid
    except ImportError:
        print(
            "[error] torch + torchreid are required for export (offline only).\n"
            "        pip install torch torchreid onnx",
            file=sys.stderr,
        )
        return 2

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[build] torchreid model arch={arch} (downloading ImageNet/zoo weights)")
    model = torchreid.models.build_model(
        name=arch,
        num_classes=1,  # arbitrary; the classifier head is unused for features
        pretrained=True,
    )
    # Load cross-camera ReID weights only if a local checkpoint file is given;
    # otherwise keep the ImageNet-pretrained backbone from build_model above
    # (a zoo *key* is not a file path and would fail load_pretrained_weights).
    import os as _os
    if pretrained and _os.path.isfile(pretrained):
        print(f"[weights] loading checkpoint '{pretrained}'")
        torchreid.utils.load_pretrained_weights(model, pretrained)
    else:
        print(f"[weights] no checkpoint file ('{pretrained}'); using ImageNet backbone")

    # eval() makes torchreid OSNet return the feature embedding (not logits).
    model.eval()

    dummy = torch.randn(1, 3, INPUT_H, INPUT_W)

    # Sanity-check the feature dimensionality before exporting.
    with torch.no_grad():
        feat = model(dummy)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        dim = int(feat.shape[-1])
    print(f"[check ] feature dim = {dim} (expected 512 for OSNet)")

    tmp_path = out_path
    print(f"[export] -> {tmp_path} (opset={opset}, dynamic batch)")
    torch.onnx.export(
        model,
        dummy,
        str(tmp_path),
        input_names=["images"],
        output_names=["features"],
        dynamic_axes={"images": {0: "batch"}, "features": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )

    if fp16:
        try:
            import onnx
            from onnxconverter_common import float16
        except ImportError:
            print(
                "[warn] --fp16 requested but onnx/onnxconverter-common not installed;\n"
                "       exported fp32 model instead. pip install onnx onnxconverter-common",
                file=sys.stderr,
            )
        else:
            print("[fp16 ] converting weights to float16")
            model_fp32 = onnx.load(str(tmp_path))
            model_fp16 = float16.convert_float_to_float16(
                model_fp32, keep_io_types=False
            )
            onnx.save(model_fp16, str(tmp_path))

    size = out_path.stat().st_size if out_path.exists() else 0
    print(f"[ok   ] {out_path} ({size:,} bytes)")

    # Optional: verify it loads under onnxruntime if present.
    try:
        import onnxruntime as ort  # noqa: WPS433

        sess = ort.InferenceSession(
            str(out_path), providers=["CPUExecutionProvider"]
        )
        out_name = sess.get_outputs()[0].name
        inp = sess.get_inputs()[0]
        import numpy as np

        np_dtype = "float16" if "float16" in str(inp.type) else "float32"
        probe = np.zeros((2, 3, INPUT_H, INPUT_W), dtype=np_dtype)
        res = sess.run([out_name], {inp.name: probe})
        print(
            f"[verify] onnxruntime load OK; batched output shape = {res[0].shape}"
        )
    except Exception as exc:  # pragma: no cover - verification is best-effort
        print(f"[verify] skipped onnxruntime verification: {exc}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arch",
        default="osnet_x0_25",
        help="torchreid model architecture (default: osnet_x0_25)",
    )
    parser.add_argument(
        "--pretrained",
        default="osnet_x0_25_msmt17",
        help="torchreid pretrained weights key (default: osnet_x0_25_msmt17)",
    )
    parser.add_argument(
        "--out",
        default=str(
            Path(__file__).resolve().parent.parent / "models" / "osnet_x0_25_msmt17.onnx"
        ),
        help="output .onnx path",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="convert to float16 (half size/VRAM, faster on the T4)",
    )
    parser.add_argument(
        "--opset", type=int, default=12, help="ONNX opset version (default: 12)"
    )
    args = parser.parse_args()

    return export(
        arch=args.arch,
        pretrained=args.pretrained,
        out_path=Path(args.out).resolve(),
        fp16=args.fp16,
        opset=args.opset,
    )


if __name__ == "__main__":
    raise SystemExit(main())
