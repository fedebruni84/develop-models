AI Look: three image-adaptive 3D LUT predictors converted to Core ML for Develop. The user picks one in the app.

**Assets** (Apple Archive, LZFSE):

| Archive | Model | Size | SHA-256 |
|---|---|---|---|
| `AILook-Zeng-FiveK-coreml.aar` | Zeng et al. (TPAMI 2020), FiveK | 2.2 MB (2210798 bytes) | `4f36c3840f98333f28c143716cb51e3df89036c1e53ac00ad64c6d7ce2c2ca39` |
| `AILook-AdaInt-FiveK-coreml.aar` | AdaInt (CVPR 2022), FiveK | 2.3 MB (2326159 bytes) | `d95be9de5d1292eb2d754885f148536cba1b6a7d77cfc7b19e9cf9319872cfa4` |
| `AILook-AdaInt-PPR10K-coreml.aar` | AdaInt (CVPR 2022), PPR10K portraits | 45.2 MB (45164801 bytes) | `fb6ed8d6c84e404f41fafb1a363c7ecebbe8273a3e657dbe5685e52c691b9625` |

**Each archive contains:**
- `<Name>.mlpackage`: ML Program in single precision. Input `image` `[1, 3, 256, 256]` (224 for PPR10K), RGB in
  [0, 1]. Outputs `lut` `[3, 33, 33, 33]` and, for AdaInt, `vertices` `[3, 33]`. Minimum macOS 15 / iOS 18; run
  it on CPU and GPU.
- `manifest.json`: input preparation, sources with their SHA-256, conversion check and self-test data.
- `LICENSE` (Apache-2.0) and `NOTICE` (authors, changes, training data).

**Training data:** MIT-Adobe FiveK and PPR10K, licensed for research use. See the model card.

**Conversion check:** maximum error against PyTorch 3.6 × 10⁻⁷ (must be below 10⁻³).

Built with `scripts/convert-ai-look.py`; the tool versions are pinned in `scripts/requirements.txt`. Details are
in [models/ai-look](https://github.com/fedebruni84/develop-models/tree/main/models/ai-look).
