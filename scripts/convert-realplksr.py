#!/usr/bin/env python3
"""Build the Super Resolution model Develop downloads (roadmap 7.5): RealPLKSR 2x, Core ML, half precision.

Source: darktable-ai `upscale-realplksr.dtmodel` (release-5.6.0), MIT, pinned by SHA-256. Its `model_x2.onnx` is
converted to PyTorch (opset 17, onnx2torch) and then to a Core ML ML Program with a fixed 1024 x 1024 input and
float16 input/output. The conversion is checked against the PyTorch model; a procedural self-test tile records the
statistics the app compares after installing the model on a device.

Output in build/models/: RealPLKSR-x2-coreml.aar (Apple Archive, LZFSE) with the .mlpackage, manifest.json and
LICENSE; it rewrites Shared/Services/SuperResolutionCatalog.swift with the archive's SHA-256, size and self-test
statistics, and prints them as a JSON line. Publish the archive as a release asset of fedebruni84/develop-models.

    python3 -m venv build/sr-venv && build/sr-venv/bin/pip install -r scripts/requirements.txt
    build/sr-venv/bin/python scripts/convert-realplksr.py
"""
import hashlib
import json
import math
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
SOURCE_URL = "https://github.com/darktable-org/darktable-ai/releases/download/release-5.6.0/upscale-realplksr.dtmodel"
SOURCE_SHA256 = "7f7b4942b2363691628f10c52bce66ff3ac67d07d4d55c3804d3f89fc9cd6ba7"
TILE = 1024
MARGIN = 32
VERSION = "1"
NAME = "RealPLKSR-x2"
LICENSE_TEXT = """MIT License

Copyright (c) 2024 Dongheon Lee

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def self_test_input() -> np.ndarray:
    """The procedural tile the app rebuilds to test an installed model (same formula in Swift)."""
    y, x = np.mgrid[0:TILE, 0:TILE].astype(np.float32)
    planes = [0.5 + 0.4 * np.sin(x * 0.05 + c) * np.cos(y * 0.03 + 2 * c) for c in range(3)]
    return np.stack(planes)[None].astype(np.float16)


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
    (services / "SuperResolutionCatalog.swift").write_text(f"""//
//  SuperResolutionCatalog.swift
//  Develop
//
//  Scritto da scripts/convert-realplksr.py: hash e dimensione dell'archivio pubblicato, statistiche della prova.

nonisolated enum SuperResolutionCatalog {{
    static let sha256 = "{sha}"
    static let size: Int64 = {size}
    static let selfTestMean: [Double] = [{mean}]
    static let selfTestSamples: [[Double]] = [
{samples},
    ]
}}
""")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = OUT / "upscale-realplksr.dtmodel"
    if not source.exists() or sha256(source) != SOURCE_SHA256:
        urllib.request.urlretrieve(SOURCE_URL, source)
    assert sha256(source) == SOURCE_SHA256, "the darktable archive is not the pinned one"
    with zipfile.ZipFile(source) as z:
        onnx_bytes = z.read("upscale-realplksr/model_x2.onnx")

    model = convert(version_converter.convert_version(onnx.load_from_string(onnx_bytes), 17)).eval()
    probe = torch.rand(1, 3, TILE, TILE, generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        traced = torch.jit.trace(model, probe)
        reference = model(probe).numpy()
    ml = ct.convert(traced, inputs=[ct.TensorType(name="input", shape=probe.shape, dtype=np.float16)],
                    outputs=[ct.TensorType(name="output", dtype=np.float16)], convert_to="mlprogram",
                    compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.iOS18)
    ml.short_description = "RealPLKSR 2x super-resolution (MIT, Dongheon Lee et al.), from darktable-ai"

    package_dir = OUT / NAME
    shutil.rmtree(package_dir, ignore_errors=True)
    package_dir.mkdir()
    package = package_dir / f"{NAME}.mlpackage"
    ml.save(str(package))

    gpu = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_AND_GPU)
    converted = gpu.predict({"input": probe.numpy().astype(np.float16)})["output"].astype(np.float32)
    error = float(np.abs(converted - reference).max())
    assert error < 0.02, f"Core ML differs from the source model by {error}"

    tested = gpu.predict({"input": self_test_input()})["output"].astype(np.float32)[0]
    self_test = {
        "formula": "0.5 + 0.4 * sin(x * 0.05 + c) * cos(y * 0.03 + 2 * c), 1024 x 1024, float16",
        "mean": [float(tested[c].mean()) for c in range(3)],
        "samples": [[c, y, x, float(tested[c, y, x])] for c, y, x in sample_points()],
        "tolerance": 0.03,
    }
    manifest = {
        "id": "realplksr-x2", "name": "RealPLKSR 2x", "version": VERSION, "scale": 2, "tile": TILE, "margin": MARGIN,
        "input": "RGB float16 [0, 1], NCHW", "computeUnits": "cpuAndGPU (the Neural Engine is inaccurate for this model)",
        "licence": {"name": "MIT", "url": "https://github.com/dslisleedh/PLKSR/blob/main/LICENSE",
                    "copyright": "Copyright (c) 2024 Dongheon Lee"},
        "source": {"model": "https://github.com/dslisleedh/PLKSR", "onnx": SOURCE_URL, "sha256": SOURCE_SHA256,
                   "paper": "https://arxiv.org/abs/2404.11848"},
        "maxErrorAgainstSource": error, "selfTest": self_test,
    }
    (package_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (package_dir / "LICENSE").write_text(LICENSE_TEXT)

    archive = OUT / f"{NAME}-coreml.aar"
    archive.unlink(missing_ok=True)
    subprocess.run(["aa", "archive", "-d", str(package_dir), "-o", str(archive), "-a", "lzfse"], check=True)
    write_catalog(sha256(archive), archive.stat().st_size, self_test)
    print(json.dumps({"archive": str(archive), "sha256": sha256(archive), "size": archive.stat().st_size,
                      "maxErrorAgainstSource": error, "selfTest": self_test}))


if __name__ == "__main__":
    main()
