# AI Look — image-adaptive 3D LUTs, Core ML

Three small networks that look at a photo and predict a 3D colour lookup table (LUT) for it. Develop uses one of
them, chosen by the user, to give a photo an automatic tone and colour look. The LUT is stored in the photo's
settings and applied on top of every other adjustment; the sliders never move. The user can change its amount or
regenerate it after editing the photo.

## Models

| | Zeng FiveK | AdaInt FiveK | AdaInt PPR10K |
|---|---|---|---|
| Method | Image-adaptive 3D LUT: a CNN weights 3 basis LUTs | AdaInt: 3 basis LUTs and learned, non-uniform sample positions | AdaInt with a ResNet-18 backbone, 5 basis LUTs |
| Paper | Zeng et al., TPAMI 2020, [arXiv:2009.14468](https://arxiv.org/abs/2009.14468) | Yang et al., CVPR 2022, [arXiv:2204.13983](https://arxiv.org/abs/2204.13983) | same |
| Code and weights | <https://github.com/HuiZeng/Image-Adaptive-3DLUT> | <https://github.com/ImCharlesY/AdaInt> | same |
| Training data | MIT-Adobe FiveK, expert C | MIT-Adobe FiveK, expert C | PPR10K (portraits), expert A |
| Good for | General photos, a stronger look | General photos, the most balanced | Portraits: brightens without oversaturating skin |
| Network input | `image`: float32 `[1, 3, 256, 256]` | `image`: float32 `[1, 3, 256, 256]` | `image`: float32 `[1, 3, 224, 224]` |
| Outputs | `lut`: `[3, 33, 33, 33]` | `lut`: `[3, 33, 33, 33]`, `vertices`: `[3, 33]` | same |
| Archive | `AILook-Zeng-FiveK-coreml.aar`, 2.2 MB | `AILook-AdaInt-FiveK-coreml.aar`, 2.3 MB | `AILook-AdaInt-PPR10K-coreml.aar`, 45.2 MB |
| Error against PyTorch | 2.4 × 10⁻⁷ | 2.4 × 10⁻⁷ | 3.6 × 10⁻⁷ |

All three: Core ML ML Program in single precision (half precision changes the LUT by up to 8/255), minimum
macOS 15 / iOS 18, run on CPU and GPU. One prediction takes 1–3 ms on an M2 Pro.

Release [`ai-look-coreml-1`](https://github.com/fedebruni84/develop-models/releases/tag/ai-look-coreml-1).
SHA-256 of each archive:

```
4f36c3840f98333f28c143716cb51e3df89036c1e53ac00ad64c6d7ce2c2ca39  AILook-Zeng-FiveK-coreml.aar
d95be9de5d1292eb2d754885f148536cba1b6a7d77cfc7b19e9cf9319872cfa4  AILook-AdaInt-FiveK-coreml.aar
fb6ed8d6c84e404f41fafb1a363c7ecebbe8273a3e657dbe5685e52c691b9625  AILook-AdaInt-PPR10K-coreml.aar
```

## Input and output

- **Input:** the photo as currently edited, rendered in sRGB with its long edge at 480 pixels (the size of the
  training photos), then resized to the network's square input with bilinear interpolation and no antialiasing
  (PyTorch `interpolate(mode="bilinear", align_corners=False)`), as in training. The aspect ratio is not kept.
  Values are RGB in [0, 1], planar.
- **LUT:** 33 × 33 × 33 entries per channel, as three planes with red varying fastest (the order of a `.cube`
  file). Values may fall slightly outside [0, 1].
- **Vertices (AdaInt):** the positions of the 33 samples along red, green and blue, rising from 0 to 1. AdaInt
  spaces them more densely where the colour changes more. The LUT must be looked up at these positions, not on a
  uniform grid: resampling it uniformly at 33 points changes some pixels by up to 18/255.
- **Applying it:** trilinear interpolation of the sRGB-encoded colour, as the authors' `lut_transform` and
  `ailut_transform`. Develop applies it after every other adjustment and mixes it with the photo by the user's
  amount.

## Conversion

`scripts/convert-ai-look.py`:

1. Downloads the authors' weights at pinned commits and checks their SHA-256.
2. Rebuilds the networks in plain PyTorch. The authors' code needs CUDA extensions only to apply the LUT, which the
   app does itself. Develop checked the rebuild against the authors' CPU kernels on 30 photos: identical outputs.
3. Converts each network to Core ML (FP32) and compares it with PyTorch on a random input. The error must be below
   10⁻³.
4. Runs a procedural self-test picture, `0.5 + 0.4 · sin(0.05x + c) · cos(0.03y + 2c)`, and records 48 LUT values
   and every fourth vertex (tolerance 2 × 10⁻³). Develop repeats the test on every device after installing a model.
5. Writes `manifest.json`, `LICENSE` and `NOTICE` next to the `.mlpackage` and packs everything with
   `aa archive -a lzfse`.

## Licences and training data

- **Code and weights:** Apache License 2.0 ([LICENSE](LICENSE)), from the authors' repositories:
  - Zeng et al.: Hui Zeng, Jianrui Cai, Lida Li, Zisheng Cao, Lei Zhang.
  - AdaInt: Canqian Yang, Meiguang Jin, Xu Jia, Yi Xu, Ying Chen; built on MMEditing.
  - Each archive carries a `NOTICE` that states the changes: the networks are re-expressed in plain PyTorch and
    converted to Core ML.
- **Training data:** MIT-Adobe FiveK and PPR10K. Their photos and expert retouches are licensed **for research
  use**. Develop is a personal, non-commercial app and uses the published weights on that basis. Anyone reusing
  these archives should check the dataset terms for their own use.

## Limitations

- The models learned the taste of one retoucher on photos straight from the camera. On a photo that is already
  edited they can overdo it: on retouched portraits the FiveK models turn skin orange. Lower the amount, or use the
  PPR10K model for portraits.
- FiveK models can close up the shadows of dark photos. PPR10K brightens strongly and can wash out bright objects.
- The look is global: the same colour gets the same change everywhere in the photo.
