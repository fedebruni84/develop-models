# RealPLKSR 2× — Core ML

Super-resolution that doubles each side of a photo (4× the pixels). Develop uses it to create a new, larger copy
of a developed photo in the Photos library. The original is never changed.

## Summary

| | |
|---|---|
| Architecture | RealPLKSR (Partial Large Kernel CNN for super-resolution), 2× |
| Authors | Dongheon Lee et al., Machine Intelligence Laboratory, University of Seoul |
| Upstream | <https://github.com/dslisleedh/PLKSR> · [arXiv:2404.11848](https://arxiv.org/abs/2404.11848) |
| Licence | MIT (see [LICENSE](LICENSE)) |
| Source file | darktable-ai `upscale-realplksr.dtmodel`, release `release-5.6.0`, member `upscale-realplksr/model_x2.onnx` |
| Source SHA-256 | `7f7b4942b2363691628f10c52bce66ff3ac67d07d4d55c3804d3f89fc9cd6ba7` (the whole `.dtmodel`) |
| Format | Core ML ML Program, half precision, minimum macOS 15 / iOS 18 |
| Input | `input`: float16, shape `[1, 3, 1024, 1024]`, RGB in [0, 1], planar (NCHW) |
| Output | `output`: float16, shape `[1, 3, 2048, 2048]`, RGB in [0, 1] |
| Compute units | CPU and GPU. The Neural Engine is not accurate enough for this model (maximum error ≈ 0.37). |
| Release | [`realplksr-x2-coreml-1`](https://github.com/fedebruni84/develop-models/releases/tag/realplksr-x2-coreml-1) |
| Archive | `RealPLKSR-x2-coreml.aar`, 13.8 MB (13802843 bytes), SHA-256 `2235ac0145c4a0c085eec9e29619ca111da3ecb01ef16c26c132bb02f9b3e39d` |

## Conversion

`scripts/convert-realplksr.py`:

1. Downloads the darktable-ai archive and checks its SHA-256.
2. Converts `model_x2.onnx` to opset 17 so that onnx2torch can read it, then to PyTorch.
3. Traces the model at 1024 × 1024 and converts it with coremltools: ML Program, FP16 compute, FP16 input and
   output.
4. Compares Core ML on the GPU with the PyTorch model on a random tile. The maximum absolute error must be
   below 0.02; the build measured 0.0078.
5. Runs the procedural self-test tile and records its per-channel means and 48 sample points (tolerance 0.03).
   The tile is `0.5 + 0.4 · sin(0.05x + c) · cos(0.03y + 2c)`, at 1024 × 1024 in float16. Develop repeats this
   test on every device after installing the model.
6. Writes `manifest.json` and `LICENSE` next to the `.mlpackage` and packs everything with
   `aa archive -a lzfse`.

## How Develop runs it

- **Base image:** the photo is developed at full resolution with the current adjustments, exactly as for
  "Save to Photos".
- **Tiles:** the image is split into 1024 × 1024 tiles that overlap by 32 pixels on each side. Only the centre
  of each result is kept, so tile edges never show.
- **Output:** an 8-bit JPEG with the original's capture metadata, creation date and location, added to Photos
  as a new photo. If the photo was opened from an album, the copy is added to that album too.
- **Limits:** the result is capped at 250 MP on Mac and 120 MP on iPad, for memory.
- **Speed:** about 0.3 MP of input per second on an M2 Pro GPU, so roughly 1.5 minutes for a 24 MP photo.

## Training data

Per the darktable-ai model card: RealPLKSR was trained with a realistic degradation pipeline (noise, blur,
compression), following common neosr practice on DF2K (DIV2K + Flickr2K). Flickr2K does not carry an explicit
open-source licence.

## Limitations

- Like any super-resolution model, it can create detail that was not in the photo, especially on fine
  repetitive texture, text and faces. It is meant for prints and crops, not for documentary accuracy.
- It works on the developed 8/16-bit image, not on raw data. Denoise first (Develop's AI Denoise) for
  high-ISO photos.
