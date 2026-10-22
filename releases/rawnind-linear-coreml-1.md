RawNIND UtNet2 linear variant (linear Rec.2020 RGB), converted to Core ML in full precision for Develop's AI Denoise on
Fujifilm X-Trans raws, Apple ProRAW and linear DNGs, and JPEG/HEIC photos.

**Licence:** GPL-3.0. The full text is in the archive and in [models/rawnind-linear/LICENSE](https://github.com/fedebruni84/develop-models/blob/main/models/rawnind-linear/LICENSE).

**Asset:** `RawNIND-linear-coreml.aar` (Apple Archive, LZFSE), 29.7 MB (29745105 bytes)

**SHA-256:**

```
49e452421f0c66e2d4395cca22ef7fa0d74ffa6b7fb60cebf104267e87ede84c
```

**Archive contents:**
- `RawNIND-linear.mlpackage`: ML Program in FP32, input and output `[1, 3, 512, 512]`. Minimum macOS 15 / iOS 18; run it
  on CPU and GPU.
- `manifest.json`: tiling, sources and self-test data.
- `LICENSE`: GNU General Public License v3.

**Source:** darktable-ai `rawdenoise-nind.dtmodel` (release-5.6.0), SHA-256
`d71b5f1e727c85a359e6f74dca9e2016c9d8fc3e2f7ac3e9b347d80ceca969af`; network `model_linear.onnx`, SHA-256
`df957efadcc152c007d5d3b0917bdff9e41c0d4a0efe56584ef30b36393cd181`. Author: Benoit Brummer (UCLouvain),
<https://github.com/trougnouf/rawnind_jddc>.

**Conversion check:** maximum error against the source network 1.3e-6 of its peak (must be below 1e-4).

Built with `scripts/convert-rawnind.py --network linear`; the tool versions are pinned in `scripts/requirements.txt`.
