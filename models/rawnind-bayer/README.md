# RawNIND UtNet2 (Bayer): Core ML

Develop's AI Denoise uses this raw denoiser on raw files from Bayer sensors. It removes noise before demosaicing and
returns a cleaned picture of the whole photo, which Develop blends in with the AI Denoise Amount slider. The original file
is never changed.

## Summary

| | |
|---|---|
| Architecture | UtNet2 (U-Net), RawNIND Bayer variant: 4 Bayer planes in, demosaiced camera RGB out |
| Author | Benoit Brummer, UCLouvain |
| Upstream | <https://github.com/trougnouf/rawnind_jddc> · [arXiv:2501.08924](https://arxiv.org/abs/2501.08924) |
| Training data | RawNIND: real photographs of the same scenes taken noisy and clean, CC BY 4.0 and CC0 (Wikimedia Commons, UCLouvain Dataverse) |
| Licence | **GPL-3.0**, see [LICENSE](LICENSE) |
| Source file | darktable-ai `rawdenoise-nind.dtmodel`, release `release-5.6.0`, SHA-256 `d71b5f1e727c85a359e6f74dca9e2016c9d8fc3e2f7ac3e9b347d80ceca969af` |
| Source network | `model_bayer.onnx`, SHA-256 `da27509dab6a2915da67e988acd86cf71f9d5bbc8d1aa0ed32933578a887b901`, 31056425 bytes |
| Format | Core ML ML Program, **full precision (FP32)**, minimum macOS 15 / iOS 18 |
| Input | `input`: float32, `[1, 4, 512, 512]`, Bayer planes `[R, G1, G2, B]` normalised as by LightCraft |
| Output | `output`: float32, `[1, 3, 1024, 1024]`, camera RGB in the model's own scale (about 1e6 times the input) |
| Compute units | CPU and GPU. The output exceeds half precision (65504), and the Neural Engine is not accurate enough. |
| Release | [`rawnind-bayer-coreml-1`](https://github.com/fedebruni84/develop-models/releases/tag/rawnind-bayer-coreml-1) |
| Archive | `RawNIND-bayer-coreml.aar`, 29.7 MB (29741512 bytes), SHA-256 `ec920fccf4d19acad84723026a8d5e1076c256f52ffe944e5aa0888ad84c1296` |

## Conversion

`scripts/convert-rawnind.py`:

1. Downloads darktable-ai's archive and checks its SHA-256, then the SHA-256 and size of `model_bayer.onnx`.
2. Converts the network to opset 17 so that onnx2torch can read it, then to PyTorch.
3. Traces it at 512 × 512 and converts it with coremltools: ML Program, FP32 compute, FP32 input and output.
4. Compares Core ML on the GPU with the PyTorch model on a random tile. The maximum error must be below 1e-4 of the
   tile's peak; the build measured 1.6e-6.
5. Runs a procedural self-test tile and records its peak, per-channel means and 48 sample points. The tile is
   `0.05 + 0.04 · sin(0.05x + c) · cos(0.03y + 2c)`, four planes of 512 × 512. Develop repeats this test on every
   device after installing the model, with a tolerance of 1e-4 of the peak.
6. Writes `manifest.json` (tiling, gain, sources, self-test) and `LICENSE` (the GPL-3.0 text) next to the `.mlpackage`,
   and packs everything with `aa archive -a lzfse`.

## How Develop runs it

Only the network runs in Core ML. Everything around it is the open-source LightCraft engine's, unchanged:

- decoding the raw file and the Bayer check;
- normalisation and tiling: tiles of 512 cells sharing 64, each answer scaled to match the input's mean;
- assembling the cleaned picture and storing it.

Checked on 12 raw files from Canon, Nikon, Sony and Panasonic (16 to 80 MP). With LightCraft's own network behind the
same tile hand-off, the cleaned picture is identical to LightCraft's. With this Core ML model it differs by at most one
step of the cleaned picture's half-precision storage (2⁻¹¹), as LightCraft's own graphics card does, and by about 2e-7 on
average. On a Mac with M2 Pro, a 24 MP photo takes about 1.8 s.

## Licence

The weights are licensed under the **GNU General Public License v3** by their author. This repository redistributes them
converted to Core ML under the same licence; the full text is in [LICENSE](LICENSE) and inside every archive. The
preferred form for modifying the model is the original network and training code, at
<https://github.com/trougnouf/rawnind_jddc>. This conversion can be reproduced with `scripts/convert-rawnind.py` from the
pinned darktable-ai file.

Develop does not include the weights. It downloads them to the device only after the user has seen this licence and
chosen Download. If you pass the model file on, the GPL applies to that copy.

## Limitations

- Bayer sensors only. Fujifilm X-Trans, Foveon and already demosaiced files (Apple ProRAW, linear DNGs) are not supported.
  darktable-ai's archive also carries a linear-RGB variant for those; it is not part of this release.
- Like any learned denoiser, at very high ISO it can smooth fine texture or render it in a way that was not in the scene.
