#!/usr/bin/env python3
"""Build the AI Denoise model Develop downloads (roadmap 7.3, build 18): RawNIND UtNet2 for Bayer sensors, Core ML, FP32.

Source: darktable-ai `rawdenoise-nind.dtmodel` (release-5.6.0), the same file LightCraft fetches, pinned by SHA-256.
Its `model_bayer.onnx` (GPL-3.0, Benoit Brummer, UCLouvain) is converted to PyTorch (opset 17, onnx2torch) and then to a
Core ML ML Program with a fixed 512 x 512 input and full precision: the network's output reaches ~1e5, beyond half
precision. The conversion is checked against the PyTorch model; a procedural self-test tile records the statistics the
app compares after installing the model on a device.

Only the network moves to Core ML: Develop keeps LightCraft's tiling (`denoise_bayer`, tile 512, overlap 64, gain
MatchMean) and its cleaned-picture files, with the same fingerprint as LightCraft's install of this `.onnx`.

Output in build/models/: RawNIND-bayer-coreml.aar (Apple Archive, LZFSE) with the .mlpackage, manifest.json and LICENSE
(GPL-3.0). Inside Develop's checkout it rewrites Shared/Services/DenoiseCatalog.swift. Publish the archive as a release
asset of fedebruni84/develop-models.

    python3 -m venv build/sr-venv && build/sr-venv/bin/pip install -r scripts/requirements.txt
    build/sr-venv/bin/python scripts/convert-rawnind.py
"""
import hashlib
import json
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import coremltools as ct
import numpy as np
import onnx
import torch
from onnx import version_converter
from onnx2torch import convert

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "models"
SOURCE_URL = "https://github.com/darktable-org/darktable-ai/releases/download/release-5.6.0/rawdenoise-nind.dtmodel"
SOURCE_SHA256 = "d71b5f1e727c85a359e6f74dca9e2016c9d8fc3e2f7ac3e9b347d80ceca969af"
ONNX_MEMBER = "rawdenoise-nind/model_bayer.onnx"
ONNX_SHA256 = "da27509dab6a2915da67e988acd86cf71f9d5bbc8d1aa0ed32933578a887b901"
ONNX_SIZE = 31_056_425
TILE = 512
OVERLAP = 64
VERSION = "1"
NAME = "RawNIND-bayer"
GPL_URL = "https://www.gnu.org/licenses/gpl-3.0.txt"
GPL_SHA256 = "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986"
TOLERANCE = 1e-4  # relative to the tile's largest output


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, path: Path, digest: str) -> None:
    if not path.exists() or sha256(path) != digest:
        urllib.request.urlretrieve(url, path)
    assert sha256(path) == digest, f"{path.name} is not the pinned file"


def self_test_input() -> np.ndarray:
    """The procedural Bayer tile the app rebuilds to test an installed model (same formula in Swift)."""
    y, x = np.mgrid[0:TILE, 0:TILE].astype(np.float32)
    planes = [0.05 + 0.04 * np.sin(x * 0.05 + c) * np.cos(y * 0.03 + 2 * c) for c in range(4)]
    return np.stack(planes)[None].astype(np.float32)


def sample_points():
    n = 2 * TILE
    return [(c, (k * 131) % n, (k * 197 + 61) % n) for k in range(16) for c in range(3)]


def write_catalog(sha: str, size: int, self_test: dict) -> None:
    """The constants the app checks the download and the installed model against (only inside Develop's checkout)."""
    services = ROOT / "Shared" / "Services"
    if not services.is_dir():
        return
    samples = ",\n".join(f"        [{c}, {y}, {x}, {v!r}]" for c, y, x, v in self_test["samples"])
    mean = ", ".join(repr(m) for m in self_test["mean"])
    (services / "DenoiseCatalog.swift").write_text(f"""//
//  DenoiseCatalog.swift
//  Develop
//
//  Scritto da scripts/convert-rawnind.py: hash e dimensione dell'archivio pubblicato, statistiche della prova.

nonisolated enum DenoiseCatalog {{
    static let sha256 = "{sha}"
    static let size: Int64 = {size}
    /// L'impronta del modello di LightCraft per gli stessi pesi: dà il nome alle foto pulite nella cache.
    static let fingerprint = "rawnind-bayer@{ONNX_SHA256}:{ONNX_SIZE}"
    static let selfTestPeak: Double = {self_test["peak"]!r}
    static let selfTestMean: [Double] = [{mean}]
    static let selfTestSamples: [[Double]] = [
{samples},
    ]
}}
""")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = OUT / "rawdenoise-nind.dtmodel"
    fetch(SOURCE_URL, source, SOURCE_SHA256)
    with zipfile.ZipFile(source) as z:
        onnx_bytes = z.read(ONNX_MEMBER)
    assert hashlib.sha256(onnx_bytes).hexdigest() == ONNX_SHA256 and len(onnx_bytes) == ONNX_SIZE
    licence = OUT / "gpl-3.0.txt"
    fetch(GPL_URL, licence, GPL_SHA256)

    model = convert(version_converter.convert_version(onnx.load_from_string(onnx_bytes), 17)).eval()
    probe = torch.rand(1, 4, TILE, TILE, generator=torch.Generator().manual_seed(7)) * 0.2
    with torch.no_grad():
        traced = torch.jit.trace(model, probe)
        reference = model(probe).numpy()
    ml = ct.convert(traced, inputs=[ct.TensorType(name="input", shape=probe.shape, dtype=np.float32)],
                    outputs=[ct.TensorType(name="output", dtype=np.float32)], convert_to="mlprogram",
                    compute_precision=ct.precision.FLOAT32, minimum_deployment_target=ct.target.iOS18)
    ml.short_description = "RawNIND UtNet2 Bayer raw denoiser (GPL-3.0, Benoit Brummer, UCLouvain), from darktable-ai"

    package_dir = OUT / NAME
    shutil.rmtree(package_dir, ignore_errors=True)
    package_dir.mkdir()
    package = package_dir / f"{NAME}.mlpackage"
    ml.save(str(package))

    gpu = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_AND_GPU)
    converted = gpu.predict({"input": probe.numpy()})["output"]
    error = float(np.abs(converted - reference).max() / np.abs(reference).max())
    assert error < TOLERANCE, f"Core ML differs from the source model by {error} of its peak"

    tested = gpu.predict({"input": self_test_input()})["output"][0]
    self_test = {
        "formula": "0.05 + 0.04 * sin(x * 0.05 + c) * cos(y * 0.03 + 2 * c), 4 x 512 x 512, float32",
        "peak": float(np.abs(tested).max()),
        "mean": [float(tested[c].mean()) for c in range(3)],
        "samples": [[c, y, x, float(tested[c, y, x])] for c, y, x in sample_points()],
        "tolerance": TOLERANCE,
    }
    manifest = {
        "id": "rawnind-bayer", "name": "RawNIND UtNet2 (Bayer)", "version": VERSION,
        "input": "Bayer tile as four planes [R, G1, G2, B], float32, [1, 4, 512, 512], LightCraft's normalisation",
        "output": "camera RGB, float32, [1, 3, 1024, 1024], in the model's own scale",
        "tiling": {"tile": TILE, "overlap": OVERLAP, "gain": {"matchMean": {"nominal": 1.0e6, "maxDeviation": 0.05}}},
        "computeUnits": "cpuAndGPU in full precision (the output exceeds half precision; the Neural Engine is not accurate enough)",
        "licence": {"name": "GPL-3.0", "url": "https://www.gnu.org/licenses/gpl-3.0.html"},
        "source": {"model": "https://github.com/trougnouf/rawnind_jddc", "paper": "https://arxiv.org/abs/2501.08924",
                   "archive": SOURCE_URL, "archiveSha256": SOURCE_SHA256, "onnx": ONNX_MEMBER, "onnxSha256": ONNX_SHA256},
        "fingerprint": f"rawnind-bayer@{ONNX_SHA256}:{ONNX_SIZE}",
        "maxErrorAgainstSource": error, "selfTest": self_test,
    }
    (package_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy(licence, package_dir / "LICENSE")

    archive = OUT / f"{NAME}-coreml.aar"
    archive.unlink(missing_ok=True)
    subprocess.run(["aa", "archive", "-d", str(package_dir), "-o", str(archive), "-a", "lzfse"], check=True)
    write_catalog(sha256(archive), archive.stat().st_size, self_test)
    print(json.dumps({"archive": str(archive), "sha256": sha256(archive), "size": archive.stat().st_size,
                      "maxErrorAgainstSource": error, "selfTestPeak": self_test["peak"]}))


if __name__ == "__main__":
    main()
