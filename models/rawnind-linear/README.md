# RawNIND UtNet2 (linear Rec.2020): Core ML

Develop's AI Denoise uses this denoiser for what the Bayer model cannot read: raw files from Fujifilm X-Trans sensors,
already demosaiced raws (Apple ProRAW, linear DNGs) and JPEG/HEIC photos. It works on demosaiced linear RGB and returns
a cleaned picture at the same size, which Develop blends in with the AI Denoise Amount slider. The original file is never
changed.

## Summary

| | |
|---|---|
| Architecture | UtNet2 (U-Net), RawNIND linear variant: 3 planes of linear Rec.2020 RGB in and out, same size |
| Author | Benoit Brummer, UCLouvain |
| Upstream | <https://github.com/trougnouf/rawnind_jddc> · [arXiv:2501.08924](https://arxiv.org/abs/2501.08924) |
| Training data | RawNIND: real photographs of the same scenes taken noisy and clean, CC BY 4.0 and CC0 (Wikimedia Commons, UCLouvain Dataverse) |
| Licence | **GPL-3.0**, see [LICENSE](LICENSE) |
| Source file | darktable-ai `rawdenoise-nind.dtmodel`, release `release-5.6.0`, SHA-256 `d71b5f1e727c85a359e6f74dca9e2016c9d8fc3e2f7ac3e9b347d80ceca969af` |
| Source network | `model_linear.onnx`, SHA-256 `df957efadcc152c007d5d3b0917bdff9e41c0d4a0efe56584ef30b36393cd181`, 31053823 bytes |
| Format | Core ML ML Program, **full precision (FP32)**, minimum macOS 15 / iOS 18 |
| Input | `input`: float32, `[1, 3, 512, 512]`, linear Rec.2020 RGB |
| Output | `output`: float32, `[1, 3, 512, 512]`, linear Rec.2020 RGB in the model's own scale (bring each tile back to its input's mean) |
| Compute units | CPU and GPU, full precision (the output reaches about 1e9). |
| Release | [`rawnind-linear-coreml-1`](https://github.com/fedebruni84/develop-models/releases/tag/rawnind-linear-coreml-1) |
| Archive | `RawNIND-linear-coreml.aar`, 29.7 MB (29745105 bytes), SHA-256 `49e452421f0c66e2d4395cca22ef7fa0d74ffa6b7fb60cebf104267e87ede84c` |

## Conversion

`scripts/convert-rawnind.py --network linear`:

1. Downloads darktable-ai's archive and checks its SHA-256, then reads `model_linear.onnx`.
2. Converts the network to opset 17 so that onnx2torch can read it, then to PyTorch.
3. Traces it at 512 × 512 and converts it with coremltools: ML Program, FP32 compute, FP32 input and output.
4. Compares Core ML on the GPU with the PyTorch model on a random tile. The maximum error must be below 1e-4 of the
   tile's peak; the build measured 1.3e-6.
5. Runs a procedural self-test tile, `0.2 + 0.15 · sin(0.05x + c) · cos(0.03y + 2c)` on three planes of 512 × 512, and
   records its peak, per-channel means and 48 sample points. Develop repeats this test on every device after installing
   the model, with a tolerance of 1e-4 of the peak.
6. Writes `manifest.json` and `LICENSE` (the GPL-3.0 text) next to the `.mlpackage` and packs everything with
   `aa archive -a lzfse`.

## How Develop runs it

Develop's own chain, chosen on measurements (darktable's preparation of this network was the starting point):

- **Raw files:** the active area is demosaiced by the LightCraft engine, white-balanced as shot in camera space and taken
  to linear Rec.2020 with the camera's matrix. After the network it goes back to camera RGB, so the engine pairs it with
  the photo exactly as a Bayer model's result. Clipped samples keep their value.
- **JPEG/HEIC:** decoded at full size into linear Rec.2020, the network's own space.
- **Tiles:** 512 × 512 with 32 pixels of mirrored context on each side; each answer is scaled to its input tile's mean,
  and the seams are feathered.

On three Fujifilm X-T1 scenes at ISO 6400 from RawNIND's held-out test set, against the ISO 200 reference: PSNR from 24.4,
24.0 and 26.7 dB to 27.4, 30.1 and 29.3 dB, with 5 to 11 times less grain in flat areas. A 16 MP photo takes about 2.3 s
on a Mac with M2 Pro.

## Licence

The weights are licensed under the **GNU General Public License v3** by their author. This repository redistributes them
converted to Core ML under the same licence; the full text is in [LICENSE](LICENSE) and inside every archive. The
preferred form for modifying the model is the original network and training code, at
<https://github.com/trougnouf/rawnind_jddc>. This conversion can be reproduced with `scripts/convert-rawnind.py --network
linear` from the pinned darktable-ai file.

Develop does not include the weights. It downloads them to the device only after the user has seen this licence and
chosen Download. If you pass the model file on, the GPL applies to that copy.

## Limitations

- JPEG and HEIC photos have already been processed and compressed by the camera or an editor: the network can do less
  there, and on heavily processed pictures it can shift tones slightly in the shadows. Develop starts JPEGs at a lower
  Amount.
- Like any learned denoiser, at very high ISO it can smooth fine texture or render it in a way that was not in the scene.
