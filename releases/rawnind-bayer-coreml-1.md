RawNIND UtNet2 for Bayer sensors (raw denoiser), converted to Core ML in full precision for Develop's AI Denoise.

**Licence:** GPL-3.0. The full text is in the archive and in [models/rawnind-bayer/LICENSE](https://github.com/fedebruni84/develop-models/blob/main/models/rawnind-bayer/LICENSE).

**Asset:** `RawNIND-bayer-coreml.aar` (Apple Archive, LZFSE), 29.7 MB (29741512 bytes)

**SHA-256:**

```
ec920fccf4d19acad84723026a8d5e1076c256f52ffe944e5aa0888ad84c1296
```

**Archive contents:**
- `RawNIND-bayer.mlpackage`: ML Program in FP32, input `[1, 4, 512, 512]` and output `[1, 3, 1024, 1024]`.
  Minimum macOS 15 / iOS 18; run it on CPU and GPU.
- `manifest.json`: tiling, sources and self-test data.
- `LICENSE`: GNU General Public License v3.

**Source:** darktable-ai `rawdenoise-nind.dtmodel` (release-5.6.0), SHA-256
`d71b5f1e727c85a359e6f74dca9e2016c9d8fc3e2f7ac3e9b347d80ceca969af`; network `model_bayer.onnx`, SHA-256
`da27509dab6a2915da67e988acd86cf71f9d5bbc8d1aa0ed32933578a887b901`. Author: Benoit Brummer (UCLouvain),
<https://github.com/trougnouf/rawnind_jddc>.

**Conversion check:** maximum error against the source network 1.6e-6 of its peak (must be below 1e-4).

Built with `scripts/convert-rawnind.py`; the tool versions are pinned in `scripts/requirements.txt`.
