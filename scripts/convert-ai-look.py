#!/usr/bin/env python3
"""Build the AI look models Develop downloads (roadmap 7.2): image-adaptive 3D LUT predictors in Core ML (FP32).

Three models, each its own archive, chosen by the user in the app:

- zeng-fivek: "Learning Image-adaptive 3D Lookup Tables" (Zeng et al., TPAMI 2020), the sRGB FiveK model.
- adaint-fivek: "AdaInt: Learning Adaptive Intervals for 3D Lookup Tables" (Yang et al., CVPR 2022), sRGB FiveK.
- adaint-ppr10k: AdaInt trained on PPR10K (portraits, expert A).

Weights: the authors' repositories (Apache-2.0 code), pinned by commit and SHA-256. They were trained on MIT-Adobe
FiveK and PPR10K, whose photos are licensed for research; the manifest and the model card say so.

The networks are rebuilt here in plain PyTorch (the authors' code needs CUDA extensions only to apply the LUT, which
the engine does) and checked against the authors' formulas: the backbone sees the photo as trained (a 480 px render
resized to 256 x 256, or 224 for the ResNet of PPR10K, bilinear without antialiasing, done by the engine bridge) and
outputs the LUT (3 x 33^3, planes red-fastest) and, for AdaInt, its sample positions (3 x 33). A procedural self-test
records outputs the app checks after installing a model.

Output in build/models/: <Name>-coreml.aar per model (Apple Archive, LZFSE) with the .mlpackage, manifest.json,
LICENSE and NOTICE; Shared/Services/AILookCatalog.swift is rewritten with hashes, sizes and self-tests.

    python3 -m venv build/sr-venv && build/sr-venv/bin/pip install -r scripts/requirements.txt
    build/sr-venv/bin/python scripts/convert-ai-look.py
"""
import hashlib
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "models"
VERSION = "1"
D = 33

ADAINT = "https://raw.githubusercontent.com/ImCharlesY/AdaInt/5678851339dff90becb09c3dec41e7214207ccbc"
ZENG = "https://raw.githubusercontent.com/HuiZeng/Image-Adaptive-3DLUT/b491f6df64a588864739a157db271e5c848e1805"
FILES = {
    "AiLUT-FiveK-sRGB.pth": (f"{ADAINT}/pretrained/AiLUT-FiveK-sRGB.pth", "bca51f280f148b1edaa71a863e6f57a78668162e2cc35fcd93b5a8c1ed81eacf"),
    "AiLUT-PPR10KA-sRGB.pth": (f"{ADAINT}/pretrained/AiLUT-PPR10KA-sRGB.pth", "b6762645e6325b91d09101bc858d628d15a7c76e2e05fe79a7a64bfed35f7b1a"),
    "AdaInt-LICENSE": (f"{ADAINT}/LICENSE", "2421dad0343e349994a824a26474ed9b616556d353248fa5cf1b371760742cbd"),
    "zeng-LUTs.pth": (f"{ZENG}/pretrained_models/sRGB/LUTs.pth", "c1bb2bc4b7239c1a7e96159f5923123ba796b1fceb0b8c3132b423ea825b821a"),
    "zeng-classifier.pth": (f"{ZENG}/pretrained_models/sRGB/classifier.pth", "bae9865395625ecae58cfe86147e521093bb1e29e7b2544e02adb238b8035021"),
    "Zeng-LICENSE": (f"{ZENG}/LICENSE", "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"),
}
DATA_NOTE = ("Trained on {data}, whose photos and expert retouches are licensed for research use. Develop uses the "
             "published weights in a personal, non-commercial app.")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name: str) -> Path:
    url, digest = FILES[name]
    path = OUT / "lut-sources" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or sha256(path) != digest:
        urllib.request.urlretrieve(url, path)
    assert sha256(path) == digest, f"{name} is not the pinned file"
    return path


# ---------------------------------------------------------------- the networks, as in the authors' code
def block(cin, cout, norm):
    layers = [nn.Conv2d(cin, cout, 3, stride=2, padding=1), nn.LeakyReLU(0.2)]
    if norm:
        layers.append(nn.InstanceNorm2d(cout, affine=True))
    return layers


class Zeng(nn.Module):
    """models.py `Classifier` (without its 256 x 256 resize, done before) weighting the three basis LUTs."""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Identity(),  # index 0 was the Upsample
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.LeakyReLU(0.2), nn.InstanceNorm2d(16, affine=True),
            *block(16, 32, True), *block(32, 64, True), *block(64, 128, True), *block(128, 128, False),
            nn.Dropout(p=0.5), nn.Conv2d(128, 3, 8, padding=0))
        luts = torch.load(fetch("zeng-LUTs.pth"), map_location="cpu")
        self.register_buffer("basis", torch.stack([luts[k]["LUT"] for k in ("0", "1", "2")]))
        missing, unexpected = self.load_state_dict(torch.load(fetch("zeng-classifier.pth"), map_location="cpu"), strict=False)
        assert missing == ["basis"] and not unexpected, (missing, unexpected)
        self.eval()

    def forward(self, x):
        w = self.model(x).view(3, 1, 1, 1, 1)
        return (w * self.basis).sum(0)


class TPAMIBackbone(nn.Sequential):
    def __init__(self):
        body = [nn.Sequential(*block(cin, cout, norm)) for cin, cout, norm in
                ((3, 16, True), (16, 32, True), (32, 64, True), (64, 128, True), (128, 128, False))]
        super().__init__(*body, nn.Dropout(p=0.5), nn.AdaptiveAvgPool2d(2))

    def forward(self, x):
        return super().forward(x).flatten(1)


class Res18Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torchvision.models.resnet18(weights=None)
        self.net.fc = nn.Identity()

    def forward(self, x):
        return self.net(x).flatten(1)


class AdaInt(nn.Module):
    """model.py `AiLUT.forward_dummy` up to the LUT and its vertices (`ailut_transform` is the engine's)."""

    def __init__(self, weights: Path, backbone: nn.Module, ranks: int):
        super().__init__()
        self.backbone = backbone
        self.weights_generator = nn.Linear(512, ranks)
        self.basis = nn.Linear(ranks, 3 * D ** 3, bias=False)
        self.intervals = nn.Linear(512, (D - 1) * 3)
        sd = torch.load(weights, map_location="cpu", weights_only=False)["state_dict"]
        rename = {"lut_generator.weights_generator.": "weights_generator.", "lut_generator.basis_luts_bank.": "basis.",
                  "adaint.intervals_generator.": "intervals."}
        mapped = {}
        for k, v in sd.items():
            for a, b in rename.items():
                if k.startswith(a):
                    k = b + k[len(a):]
            mapped[k] = v
        missing, unexpected = self.load_state_dict(mapped, strict=False)
        assert not unexpected and all("num_batches" in m for m in missing), (missing, unexpected)
        self.eval()

    def forward(self, x):
        codes = self.backbone(x)
        lut = self.basis(self.weights_generator(codes)).view(3, D, D, D)
        intervals = self.intervals(codes).view(3, D - 1).softmax(-1)
        return lut, F.pad(intervals.cumsum(-1), (1, 0), "constant", 0.0)


MODELS = [
    {
        "id": "zeng-fivek", "name": "Zeng FiveK", "archive": "AILook-Zeng-FiveK", "res": 256, "adaptive": False,
        "build": lambda: Zeng(), "licence": "Zeng-LICENSE", "weights": ["zeng-classifier.pth", "zeng-LUTs.pth"],
        "copyright": "Copyright (c) Hui Zeng, Jianrui Cai, Lida Li, Zisheng Cao, Lei Zhang",
        "source": "https://github.com/HuiZeng/Image-Adaptive-3DLUT", "paper": "https://arxiv.org/abs/2009.14468",
        "data": "MIT-Adobe FiveK (expert C)",
    },
    {
        "id": "adaint-fivek", "name": "AdaInt FiveK", "archive": "AILook-AdaInt-FiveK", "res": 256, "adaptive": True,
        "build": lambda: AdaInt(fetch("AiLUT-FiveK-sRGB.pth"), TPAMIBackbone(), 3), "licence": "AdaInt-LICENSE",
        "weights": ["AiLUT-FiveK-sRGB.pth"],
        "copyright": "Copyright (c) Canqian Yang, Meiguang Jin, Xu Jia, Yi Xu, Ying Chen; MMEditing Authors",
        "source": "https://github.com/ImCharlesY/AdaInt", "paper": "https://arxiv.org/abs/2204.13983",
        "data": "MIT-Adobe FiveK (expert C)",
    },
    {
        "id": "adaint-ppr10k", "name": "AdaInt PPR10K", "archive": "AILook-AdaInt-PPR10K", "res": 224, "adaptive": True,
        "build": lambda: AdaInt(fetch("AiLUT-PPR10KA-sRGB.pth"), Res18Backbone(), 5), "licence": "AdaInt-LICENSE",
        "weights": ["AiLUT-PPR10KA-sRGB.pth"],
        "copyright": "Copyright (c) Canqian Yang, Meiguang Jin, Xu Jia, Yi Xu, Ying Chen; MMEditing Authors",
        "source": "https://github.com/ImCharlesY/AdaInt", "paper": "https://arxiv.org/abs/2204.13983",
        "data": "PPR10K (portraits, expert A)",
    },
]


def self_test_input(res: int) -> np.ndarray:
    """The procedural picture the app rebuilds to test an installed model (same formula in Swift)."""
    y, x = np.mgrid[0:res, 0:res].astype(np.float32)
    planes = [0.5 + 0.4 * np.sin(x * 0.05 + c) * np.cos(y * 0.03 + 2 * c) for c in range(3)]
    return np.stack(planes)[None].astype(np.float32)


def lut_samples():
    n = 3 * D ** 3
    return [(k * 7919 + 13) % n for k in range(48)]


def convert(spec: dict) -> dict:
    model = spec["build"]()
    res = spec["res"]
    probe = torch.rand(1, 3, res, res, generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        traced = torch.jit.trace(model, probe)
        reference = model(probe)
    outputs = ["lut", "vertices"] if spec["adaptive"] else ["lut"]
    ml = ct.convert(traced, inputs=[ct.TensorType(name="image", shape=probe.shape, dtype=np.float32)],
                    outputs=[ct.TensorType(name=o, dtype=np.float32) for o in outputs], convert_to="mlprogram",
                    compute_precision=ct.precision.FLOAT32, minimum_deployment_target=ct.target.iOS18)
    ml.short_description = f"{spec['name']}: image-adaptive 3D LUT ({spec['source']})"

    name = f"{spec['archive']}"
    package_dir = OUT / name
    shutil.rmtree(package_dir, ignore_errors=True)
    package_dir.mkdir(parents=True)
    package = package_dir / f"{name}.mlpackage"
    ml.save(str(package))

    gpu = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_AND_GPU)
    got = gpu.predict({"image": probe.numpy()})
    ref = reference if isinstance(reference, tuple) else (reference,)
    error = max(float(np.abs(got[o].reshape(-1) - r.numpy().reshape(-1)).max()) for o, r in zip(outputs, ref))
    assert error < 1e-3, f"Core ML differs from PyTorch by {error}"

    tested = gpu.predict({"image": self_test_input(res)})
    lut = tested["lut"].reshape(-1)
    self_test = {
        "formula": f"0.5 + 0.4 * sin(x * 0.05 + c) * cos(y * 0.03 + 2 * c), {res} x {res}, float32",
        "lut": [[i, float(lut[i])] for i in lut_samples()],
        "vertices": [float(v) for v in tested["vertices"].reshape(-1)[::4]] if spec["adaptive"] else [],
        "tolerance": 2e-3,
    }
    manifest = {
        "id": spec["id"], "name": spec["name"], "version": VERSION, "lutSize": D, "inputSize": res,
        "input": f"RGB float32 [0, 1], NCHW 1 x 3 x {res} x {res}: a 480 px sRGB render resized bilinearly without "
                 "antialiasing (PyTorch interpolate, align_corners=False)",
        "outputs": {"lut": f"3 x {D}^3, planes, red fastest", **({"vertices": f"3 x {D}, rising from 0 to 1"} if spec["adaptive"] else {})},
        "computeUnits": "cpuAndGPU",
        "licence": {"name": "Apache-2.0", "copyright": spec["copyright"]},
        "trainingData": DATA_NOTE.format(data=spec["data"]),
        "source": {"code": spec["source"], "paper": spec["paper"], "weights": {FILES[k][0]: FILES[k][1] for k in spec["weights"]}},
        "maxErrorAgainstPyTorch": error, "selfTest": self_test,
    }
    (package_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy(fetch(spec["licence"]), package_dir / "LICENSE")
    (package_dir / "NOTICE").write_text(
        f"{spec['name']} for Develop, converted to Core ML from the authors' published weights.\n"
        f"{spec['copyright']}. Licensed under the Apache License 2.0 (LICENSE).\n"
        f"Source: {spec['source']} ({spec['paper']}).\n"
        f"{DATA_NOTE.format(data=spec['data'])}\n"
        "Changes: the network is re-expressed in plain PyTorch without the CUDA LUT transform and converted to Core ML.\n")

    archive = OUT / f"{name}-coreml.aar"
    archive.unlink(missing_ok=True)
    subprocess.run(["aa", "archive", "-d", str(package_dir), "-o", str(archive), "-a", "lzfse"], check=True)
    return {"id": spec["id"], "archive": str(archive), "sha256": sha256(archive), "size": archive.stat().st_size,
            "maxErrorAgainstPyTorch": error, "selfTest": self_test}


def write_catalog(results: list) -> None:
    """The constants the app checks downloads and installed models against (only inside Develop's checkout)."""
    services = ROOT / "Shared" / "Services"
    if not services.is_dir():
        return
    entries = []
    for r in results:
        lut = ", ".join(f"[{i}, {v!r}]" for i, v in r["selfTest"]["lut"])
        verts = ", ".join(repr(v) for v in r["selfTest"]["vertices"])
        entries.append(f"""        "{r['id']}": Entry(
            sha256: "{r['sha256']}",
            size: {r['size']},
            selfTestLUT: [{lut}],
            selfTestVertices: [{verts}]
        ),""")
    body = "\n".join(entries)
    (services / "AILookCatalog.swift").write_text(f"""//
//  AILookCatalog.swift
//  Develop
//
//  Scritto da scripts/convert-ai-look.py: hash e dimensione degli archivi pubblicati, risultati della prova.

nonisolated enum AILookCatalog {{
    struct Entry: Sendable {{
        let sha256: String
        let size: Int64
        /// Campioni `[indice, valore]` del LUT (piani, rosso più veloce) e un vertice su quattro, sulla foto procedurale.
        let selfTestLUT: [[Double]]
        let selfTestVertices: [Double]
    }}

    static let entries: [String: Entry] = [
{body}
    ]
}}
""")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = [convert(spec) for spec in MODELS]
    write_catalog(results)
    for r in results:
        print(json.dumps({k: r[k] for k in ("id", "archive", "sha256", "size", "maxErrorAgainstPyTorch")}))


if __name__ == "__main__":
    main()
