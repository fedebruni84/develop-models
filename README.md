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
| [RealPLKSR 2×](models/realplksr-x2/README.md) | Super-resolution, 2× per side | [`realplksr-x2-coreml-1`](https://github.com/fedebruni84/develop-models/releases/tag/realplksr-x2-coreml-1) | `RealPLKSR-x2-coreml.aar` | 13.8 MB (13802843 bytes) | [MIT](models/realplksr-x2/LICENSE) |

SHA-256 of `RealPLKSR-x2-coreml.aar`:

```
2235ac0145c4a0c085eec9e29619ca111da3ecb01ef16c26c132bb02f9b3e39d
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
```

The script writes `build/models/RealPLKSR-x2-coreml.aar` and prints its SHA-256, its size, the maximum error
against the source model and the self-test statistics. A rebuild is expected to give the same weights and
results with the pinned versions. The archive's bytes may still differ from the published one, because the
package metadata includes timestamps. Check a rebuild against `manifest.json`, not against the hash.

## Licences and attribution

- **Scripts and documentation in this repository:** [MIT](LICENSE).
- **Each model:** keeps its own licence, in `models/<model>/LICENSE`. That file is also copied into every
  archive.

**RealPLKSR**
- Authors: Dongheon Lee et al., Machine Intelligence Laboratory, University of Seoul.
- Code: <https://github.com/dslisleedh/PLKSR>
- Paper: [arXiv:2404.11848](https://arxiv.org/abs/2404.11848)
- Licence: MIT.
- The ONNX export these archives are converted from was published by
  [darktable-ai](https://github.com/darktable-org/darktable-ai) (release `release-5.6.0`).

This repository is not affiliated with the model authors, darktable, Apple or Adobe. The models are provided
"as is", without warranty, as stated in their licences.
