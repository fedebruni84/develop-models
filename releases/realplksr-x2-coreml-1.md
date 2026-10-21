RealPLKSR 2× super-resolution, converted to Core ML for Develop.

**Asset:** `RealPLKSR-x2-coreml.aar` (Apple Archive, LZFSE), 13.8 MB (13802843 bytes)

**SHA-256:**

```
2235ac0145c4a0c085eec9e29619ca111da3ecb01ef16c26c132bb02f9b3e39d
```

**Archive contents:**
- `RealPLKSR-x2.mlpackage`: ML Program in half precision, input `[1, 3, 1024, 1024]` and output
  `[1, 3, 2048, 2048]`, RGB in [0, 1]. Minimum macOS 15 / iOS 18; run it on CPU and GPU.
- `manifest.json`: source, conversion and self-test data.
- `LICENSE`: MIT, Copyright (c) 2024 Dongheon Lee.

**Source:** darktable-ai `upscale-realplksr.dtmodel` (release-5.6.0), SHA-256
`7f7b4942b2363691628f10c52bce66ff3ac67d07d4d55c3804d3f89fc9cd6ba7`, member `model_x2.onnx`.

**Conversion check:** maximum error against the source model 0.0078 (must be below 0.02).

Built with `scripts/convert-realplksr.py`; the tool versions are pinned in `scripts/requirements.txt`. Details
are in [models/realplksr-x2](https://github.com/fedebruni84/develop-models/tree/main/models/realplksr-x2).
