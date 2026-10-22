# develop-models

Machine-learning models for **Develop**, a personal photo-development app for Mac and iPad. The app does not
bundle these models. It downloads one from this repository's
[releases](https://github.com/fedebruni84/develop-models/releases) only when the user asks for it, after showing
the model's licence.

The repository holds:

- the scripts that build each model from its published upstream source;
- the documentation and the licence of each model.

The model files themselves are **release assets**, never committed to git.

## Models

| Model | Task | Release | Archive | Size | Licence |
|---|---|---|---|---|---|
| [RawNIND UtNet2 (Bayer)](models/rawnind-bayer/README.md) | Raw denoising (AI Denoise) | [`rawnind-bayer-coreml-1`](https://github.com/fedebruni84/develop-models/releases/tag/rawnind-bayer-coreml-1) | `RawNIND-bayer-coreml.aar` | 29.7 MB (29741512 bytes) | [GPL-3.0](models/rawnind-bayer/LICENSE) |
| [RealPLKSR 2×](models/realplksr-x2/README.md) | Super-resolution, 2× per side | [`realplksr-x2-coreml-1`](https://github.com/fedebruni84/develop-models/releases/tag/realplksr-x2-coreml-1) | `RealPLKSR-x2-coreml.aar` | 13.8 MB (13802843 bytes) | [MIT](models/realplksr-x2/LICENSE) |
| [AI Look (3 models)](models/ai-look/README.md) | Automatic look: a 3D LUT predicted for each photo | [`ai-look-coreml-1`](https://github.com/fedebruni84/develop-models/releases/tag/ai-look-coreml-1) | `AILook-Zeng-FiveK-coreml.aar`, `AILook-AdaInt-FiveK-coreml.aar`, `AILook-AdaInt-PPR10K-coreml.aar` | 2.2 MB, 2.3 MB, 45.2 MB | [Apache-2.0](models/ai-look/LICENSE), research-only training data |

SHA-256 of each archive:

```
ec920fccf4d19acad84723026a8d5e1076c256f52ffe944e5aa0888ad84c1296  RawNIND-bayer-coreml.aar
2235ac0145c4a0c085eec9e29619ca111da3ecb01ef16c26c132bb02f9b3e39d  RealPLKSR-x2-coreml.aar
4f36c3840f98333f28c143716cb51e3df89036c1e53ac00ad64c6d7ce2c2ca39  AILook-Zeng-FiveK-coreml.aar
d95be9de5d1292eb2d754885f148536cba1b6a7d77cfc7b19e9cf9319872cfa4  AILook-AdaInt-FiveK-coreml.aar
fb6ed8d6c84e404f41fafb1a363c7ecebbe8273a3e657dbe5685e52c691b9625  AILook-AdaInt-PPR10K-coreml.aar
```

## How Develop uses a model

1. **Licence first.** Nothing is downloaded until the user has seen the model's licence and chosen Download.
   The download can start from Settings ▸ AI Features or at the model's first use.
2. **Pinned file.** The app knows the exact URL, size and SHA-256 of each archive. A file with a different hash
   is rejected and deleted.
3. **Checked on the device.** The archive is extracted with Apple Archive and compiled by Core ML for that
   device. It then runs a fixed self-test tile and must match the statistics recorded at conversion time.
   If any step fails, nothing is installed.
4. **Local only.** Photos are processed entirely on the device. Nothing is uploaded: the only network request is
   the download of the archive.
5. **Removable.** Models live in Application Support, are excluded from backups, and can be removed from Settings.

## Archive format

Each archive is an [Apple Archive](https://developer.apple.com/documentation/applearchive) compressed with LZFSE
(`.aar`). macOS and iOS can extract it with system frameworks only. The archive contains:

| File | Content |
|---|---|
| `<Name>.mlpackage` | Core ML ML Program |
| `manifest.json` | Input/output description, tile and margin, compute units, licence, source with its SHA-256, error against the source model, self-test statistics |
| `LICENSE` | The model's licence |
| `NOTICE` | AI Look only: authors, changes and training data |

To inspect an archive on a Mac:

```bash
shasum -a 256 RealPLKSR-x2-coreml.aar
aa extract -i RealPLKSR-x2-coreml.aar -d RealPLKSR-x2
cat RealPLKSR-x2/manifest.json
```

## Rebuilding a model

The conversion is reproducible. It starts from the upstream file pinned by SHA-256, and every Python
dependency is pinned. Requirements: macOS on Apple silicon, Python 3.13, and about 16 GB of RAM. The PyTorch
reference pass at 1024 × 1024 on the CPU takes a while.

```bash
python3 -m venv build/sr-venv
build/sr-venv/bin/pip install -r scripts/requirements.txt
build/sr-venv/bin/python scripts/convert-realplksr.py
build/sr-venv/bin/python scripts/convert-rawnind.py
build/sr-venv/bin/python scripts/convert-ai-look.py
```

Each script writes its archive into `build/models/` and prints its SHA-256, its size, the maximum error against the
source model and the self-test statistics. A rebuild is expected to give the same weights and
results with the pinned versions. The archive's bytes may still differ from the published one, because the
package metadata includes timestamps. Check a rebuild against `manifest.json`, not against the hash.

## Licences and attribution

- **Scripts and documentation in this repository:** [MIT](LICENSE).
- **Each model:** keeps its own licence, in `models/<model>/LICENSE`. That file is also copied into every
  archive.

**RawNIND UtNet2**
- Author: Benoit Brummer, UCLouvain.
- Code and training: <https://github.com/trougnouf/rawnind_jddc>
- Paper: [arXiv:2501.08924](https://arxiv.org/abs/2501.08924)
- Licence: **GPL-3.0**. The weights are redistributed converted to Core ML under the same licence, with its full text.
  The preferred form for modifying them is the upstream network and training code.
- Training data: RawNIND, CC BY 4.0 and CC0.
- The ONNX export these archives are converted from was published by
  [darktable-ai](https://github.com/darktable-org/darktable-ai) (release `release-5.6.0`).

**RealPLKSR**
- Authors: Dongheon Lee et al., Machine Intelligence Laboratory, University of Seoul.
- Code: <https://github.com/dslisleedh/PLKSR>
- Paper: [arXiv:2404.11848](https://arxiv.org/abs/2404.11848)
- Licence: MIT.
- The ONNX export these archives are converted from was published by
  [darktable-ai](https://github.com/darktable-org/darktable-ai) (release `release-5.6.0`).

**AI Look (Zeng et al.; AdaInt)**
- Authors: Hui Zeng, Jianrui Cai, Lida Li, Zisheng Cao, Lei Zhang (TPAMI 2020); Canqian Yang, Meiguang Jin, Xu Jia,
  Yi Xu, Ying Chen (CVPR 2022).
- Code and weights: <https://github.com/HuiZeng/Image-Adaptive-3DLUT>, <https://github.com/ImCharlesY/AdaInt>
- Licence: **Apache-2.0**.
- Training data: MIT-Adobe FiveK and PPR10K, licensed for research use (see the model card).

This repository is not affiliated with the model authors, darktable, Apple or Adobe. The models are provided
"as is", without warranty, as stated in their licences.
