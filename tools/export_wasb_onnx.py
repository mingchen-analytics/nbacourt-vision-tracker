"""Export the WASB basketball HRNet to ONNX in a clean process (torch only, no cv2/av).

Usage: python tools/export_wasb_onnx.py <WASB-SBDT clone> <wasb_basketball_best.pth.tar>
Keep the clone and the checkpoint outside this repository (weights license not stated).
"""
import importlib.util, sys
from pathlib import Path
import torch, yaml
if len(sys.argv) != 3:
    sys.exit(__doc__)
repo = Path(sys.argv[1]); w = Path(sys.argv[2])
out = Path("models/onnx/wasb_basketball_288x512.onnx"); out.parent.mkdir(parents=True, exist_ok=True)
class A(dict):
    def __getattr__(self, k): return self[k]
def wrap(x):
    return A({k: wrap(v) for k, v in x.items()}) if isinstance(x, dict) else [wrap(v) for v in x] if isinstance(x, list) else x
cfg = wrap(yaml.safe_load((repo / "src/configs/model/wasb.yaml").read_text()))
spec = importlib.util.spec_from_file_location("wasb_hrnet", repo / "src/models/hrnet.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
net = m.HRNet(cfg)
sd = torch.load(w, map_location="cpu", weights_only=False); sd = sd.get("model_state_dict", sd)
print(net.load_state_dict({k.replace("module.", "", 1): v for k, v in sd.items()}, strict=True))
net.eval()
class Wrap(torch.nn.Module):
    def __init__(s, n): super().__init__(); s.n = n
    def forward(s, x):
        y = s.n(x); y = y[0] if isinstance(y, dict) else y
        return torch.sigmoid(y)
x = torch.rand(1, 9, 288, 512)
with torch.no_grad(): ref = Wrap(net)(x).numpy()
torch.onnx.export(Wrap(net), x, str(out), input_names=["frames"], output_names=["heatmaps"], opset_version=17, dynamo=False)
import onnxruntime as ort
y = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"]).run(None, {"frames": x.numpy()})[0]
print("exported", out, "max abs diff vs torch", float(abs(y - ref).max()), "shape", y.shape)
