               
import os
import shutil
import subprocess
from pathlib import Path

import torch

from .model import EXPORT_OUTPUTS, ExportWrapper, load_model


def export_model(model_path="weights/Aster.pt", out="export/Aster.onnx", mnn=None, opset=13, biz_code="Aster"):
    net, _ = load_model(model_path)
    model = ExportWrapper(net).eval()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    x = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, x, out, input_names=["image"], output_names=EXPORT_OUTPUTS,
                      opset_version=opset, do_constant_folding=True, dynamo=False)

    print(f"onnx: {out}")

    if mnn:
        subprocess.run(["mnnconvert", "-f", "ONNX", "--modelFile", out, "--MNNModel", mnn,
                        "--bizCode", biz_code],
                       check=True, stdout=subprocess.DEVNULL)
        print(f"mnn: {mnn}")


def install_model(model_path, target_dir):
    model_path, target_dir = Path(model_path), Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"Aster{model_path.suffix.lower()}"
    shutil.copyfile(model_path, target)
    print(f"installed: {target}")
