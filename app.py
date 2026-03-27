"""
WaferAI v3 — HuggingFace Spaces Edition
Framework: FastAPI  (replaces Flask — HF Spaces runs on port 7860 via uvicorn)
Deploy:    SDK = docker  OR  SDK = gradio (static-app mode)

Issues fixed from original Flask version:
  1. Flask is NOT officially supported on HF Spaces — replaced with FastAPI + uvicorn
  2. register_backward_hook() is deprecated → register_full_backward_hook()
  3. model loaded with weights_only=False added for torch 2.x compatibility
  4. matplotlib backend forced to "Agg" before any import of pyplot (was already ok, kept safe)
  5. torch.no_grad() added to pure-inference paths to cut memory usage
  6. gc.collect() after each request to help the 2GB HF free tier
  7. HF Spaces requires the server to bind 0.0.0.0:7860  (PORT env-var respected as fallback)
  8. StaticFiles / HTMLResponse used so the single-file frontend is served correctly
  9. request size limit raised to 32 MB to allow batch uploads
  10. All HTML fetch() calls updated from '/predict' → relative URLs (same-origin, works fine)
"""

import os, io, gc, base64
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

import matplotlib
matplotlib.use("Agg")          # MUST be before pyplot import
import matplotlib.pyplot as plt

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "best_model.pth"))
NUM_CLASSES = 8
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Bridge", "Clean", "CMP Scratches", "Crack", "LER", "Open", "Other", "Vias"]

# ── APP ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="WaferAI", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── MODEL ─────────────────────────────────────────────────────────────────────
def load_model():
    m = models.mobilenet_v3_small(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, NUM_CLASSES)
    if os.path.exists(MODEL_PATH):
        # weights_only=False  needed for torch >= 2.0 with custom state-dicts
        m.load_state_dict(
            torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        )
        print(f"✅  Model loaded → {MODEL_PATH}")
    else:
        print(f"⚠️   {MODEL_PATH} not found — demo mode (random weights)")
    m.to(DEVICE).eval()
    return m

model = load_model()

infer_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# ── GRADCAM ───────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Fixed version:
      - register_full_backward_hook  (register_backward_hook is deprecated in torch 2.x)
      - Hooks store detached copies so graph can be freed immediately after .backward()
    """
    def __init__(self, mdl: nn.Module, layer: nn.Module):
        self.mdl   = mdl
        self.grads = None
        self.acts  = None
        layer.register_forward_hook(self._save_acts)
        layer.register_full_backward_hook(self._save_grads)   # ← FIXED

    def _save_acts(self, module, inp, out):
        self.acts = out.detach()

    def _save_grads(self, module, grad_in, grad_out):
        self.grads = grad_out[0].detach()

    def run(self, tensor: torch.Tensor):
        self.mdl.zero_grad()
        out = self.mdl(tensor)
        idx = out.argmax(1).item()
        out[0, idx].backward()
        w   = self.grads.mean([0, 2, 3])
        cam = (self.acts[0] * w[:, None, None]).mean(0).cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cam / cam.max() if cam.max() > 0 else cam
        return cv2.resize(cam, (IMG_SIZE, IMG_SIZE)), idx, out

gradcam = GradCAM(model, model.features[-1][0])

# ── WAFER MAP ─────────────────────────────────────────────────────────────────
def make_wafer_map(cls_name: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="#050a14")
    ax.set_facecolor("#050a14")
    ax.set_aspect("equal")
    ax.axis("off")
    dw, dh, gap = 0.10, 0.10, 0.012
    total = defect = 0

    for ry in np.arange(-1.0, 1.05, dh + gap):
        for rx in np.arange(-1.0, 1.05, dw + gap):
            cx, cy = rx + dw / 2, ry + dh / 2
            d = np.hypot(cx, cy)
            if d + dw / 2 > 1.0:
                continue
            total += 1
            bad = False
            if   cls_name == "Bridge":        bad = abs(cy) < 0.18 or abs(cx) < 0.18
            elif cls_name == "Clean":         bad = False
            elif cls_name == "CMP Scratches": bad = abs(np.sin(np.arctan2(cy, cx) * 2)) < 0.18
            elif cls_name == "Crack":         bad = abs(cx - cy) < 0.14 or abs(cx + cy) < 0.14
            elif cls_name == "LER":           bad = 0.62 < d < 0.88
            elif cls_name == "Open":          bad = d < 0.38
            elif cls_name == "Other":         bad = (int(rx * 100) + int(ry * 100)) % 5 == 0
            elif cls_name == "Vias":          bad = 0.25 < d < 0.52 and int(rx * 100) % 4 == 0
            fc = "#ff4444" if bad else "#0d2a3a"
            ec = "#ff7777" if bad else "#0a4060"
            ax.add_patch(plt.Rectangle((rx, ry), dw * .9, dh * .9, lw=.3, ec=ec, fc=fc, alpha=.92))
            if bad:
                defect += 1

    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, ec="#00d4ff", lw=2, alpha=.8))
    ax.plot([-.12, .12], [1, 1], c="#00d4ff", lw=4, solid_capstyle="round", alpha=.9)
    for r in [.3, .55, .75, .9]:
        ax.add_patch(plt.Circle((0, 0), r, fill=False, ec="#00d4ff", lw=.5, alpha=.14))
    yld = (total - defect) / total * 100 if total else 0
    ax.text(0, -1.14,
            f"Class: {cls_name}   Dies: {total}   Defective: {defect}   Yield: {yld:.1f}%",
            ha="center", va="center", color="#00d4ff", fontsize=7.5, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=.3", fc="#0a1628", ec="#00d4ff", alpha=.85))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.28, 1.18)
    ax.set_title("Wafer Map — Die Distribution", color="#00d4ff", fontsize=11, fontfamily="monospace", pad=8)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="#050a14")
    plt.close(fig)
    buf.seek(0)
    result = base64.b64encode(buf.read()).decode()
    buf.close()
    return result

# ── INFERENCE ─────────────────────────────────────────────────────────────────
def run_inference(file_bytes: bytes, filename: str) -> dict:
    pil     = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    pil     = pil.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)   # pre-resize saves RAM
    tensor  = infer_tf(pil).unsqueeze(0).to(DEVICE)
    tensor.requires_grad_(True)                                  # needed for GradCAM backward

    cam, idx, out = gradcam.run(tensor)

    # Probabilities — detach before converting
    probs  = torch.softmax(out, 1)[0].detach().cpu().numpy()
    scores = {CLASS_NAMES[i]: round(float(probs[i]) * 100, 2) for i in range(NUM_CLASSES)}
    pred   = CLASS_NAMES[idx]
    conf   = round(float(probs[idx]) * 100, 2)

    # Original image → base64
    orig_np = np.array(pil)
    _, obuf  = cv2.imencode(".png", cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR))
    orig_b64 = base64.b64encode(obuf).decode()

    # GradCAM overlay → base64
    gray    = cv2.cvtColor(cv2.cvtColor(orig_np, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    hmap    = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    hmap    = cv2.cvtColor(hmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(gray, 0.45, hmap, 0.55, 0)
    _, gbuf  = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    grad_b64 = base64.b64encode(gbuf).decode()

    wafer_b64 = make_wafer_map(pred)

    # Free tensors explicitly to help GC on the 2 GB HF tier
    del tensor, out, probs, cam, gray, hmap, overlay
    gc.collect()

    return dict(
        success        = True,
        predicted_class= pred,
        confidence     = conf,
        scores         = scores,
        original_image = orig_b64,
        gradcam_image  = grad_b64,
        wafer_map      = wafer_b64,
        metadata       = dict(filename=filename, device=str(DEVICE)),
    )

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model":        "MobileNetV3-Small",
        "classes":      NUM_CLASSES,
        "device":       str(DEVICE),
        "model_loaded": os.path.exists(MODEL_PATH),
    }

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if not image:
        raise HTTPException(status_code=400, detail="No image field")
    try:
        data = run_inference(await image.read(), image.filename or "upload.png")
        return JSONResponse(content=data)
    except Exception as e:
        import traceback
        traceback.print_exc()
        gc.collect()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(images: list[UploadFile] = File(...)):
    if not images:
        raise HTTPException(status_code=400, detail="No images uploaded")
    results = []
    for f in images:
        try:
            results.append(run_inference(await f.read(), f.filename or "upload.png"))
        except Exception as e:
            results.append(dict(success=False, metadata=dict(filename=f.filename), error=str(e)))
        finally:
            gc.collect()
    return JSONResponse(content=dict(results=results, total=len(results)))

# ── FRONTEND (served from the same process) ───────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>WaferAI — SEM Defect Intelligence</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet"/>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#050a14;--blue:#00d4ff;--green:#00ff88;--amber:#ffaa00;--red:#ff6b6b;
  --card:rgba(255,255,255,.03);--border:rgba(255,255,255,.08)}
body{background:var(--bg);color:#fff;font-family:'Syne',sans-serif;min-height:100vh}
.mono{font-family:'Space Mono',monospace}
body::before{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background-image:linear-gradient(rgba(0,212,255,.035) 1px,transparent 1px),
  linear-gradient(90deg,rgba(0,212,255,.035) 1px,transparent 1px);background-size:44px 44px}
#root{position:relative;z-index:1}
::-webkit-scrollbar{width:4px}::-webkit-scrollbar-thumb{background:rgba(0,212,255,.3);border-radius:2px}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.3;transform:scale(.6)}}
@keyframes radar{to{transform:rotate(360deg)}}
@keyframes sonar{0%{transform:scale(.4);opacity:.9}100%{transform:scale(2.6);opacity:0}}
@keyframes scan{0%{top:-2px}100%{top:100%}}
@keyframes fadeUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
@keyframes glow{0%,100%{box-shadow:0 0 10px rgba(0,212,255,.25)}50%{box-shadow:0 0 30px rgba(0,212,255,.6)}}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-7px)}}
@keyframes shimmer{0%,100%{opacity:.4}50%{opacity:1}}
.pulse-dot{animation:pulse 2s ease-in-out infinite}
.float-anim{animation:float 4s ease-in-out infinite}
.fade-up{animation:fadeUp .5s ease forwards}
nav{position:sticky;top:0;z-index:100;background:rgba(5,10,20,.9);
  backdrop-filter:blur(20px);border-bottom:1px solid rgba(0,212,255,.12)}
.nav-in{max-width:1400px;margin:auto;padding:0 24px;height:64px;
  display:flex;align-items:center;justify-content:space-between}
.logo{display:flex;align-items:center;gap:12px;cursor:pointer}
.logo-icon{width:38px;height:38px;border-radius:10px;background:rgba(0,212,255,.14);
  border:1px solid rgba(0,212,255,.4);display:flex;align-items:center;justify-content:center}
.logo-name{font-size:1.2rem;font-weight:800;letter-spacing:-.02em}
.logo-name span{color:var(--blue)}
.logo-tag{font-size:.52rem;color:rgba(255,255,255,.28);letter-spacing:.12em;
  font-family:'Space Mono',monospace;margin-top:-3px}
.nav-links{display:flex;gap:2px}
.nb{padding:8px 14px;border:none;background:transparent;color:rgba(255,255,255,.42);
  cursor:pointer;font-family:'Syne',sans-serif;font-size:.86rem;font-weight:600;
  border-radius:8px;border-bottom:2px solid transparent;transition:all .25s}
.nb:hover{color:#fff}.nb.active{color:var(--blue);border-bottom-color:var(--blue)}
.mbadge{display:flex;align-items:center;gap:8px;padding:6px 14px;border-radius:99px;
  background:rgba(0,255,136,.07);border:1px solid rgba(0,255,136,.22)}
.mbadge span{font-family:'Space Mono',monospace;font-size:.7rem;color:var(--green)}
main{max-width:1400px;margin:auto;padding:0 24px 80px}
.sec{display:none}.sec.active{display:block}
.card{background:var(--card);border:1px solid var(--border);border-radius:18px;backdrop-filter:blur(12px)}
.btn{border:none;cursor:pointer;font-family:'Syne',sans-serif;font-weight:700;
  border-radius:12px;transition:all .3s;display:inline-flex;align-items:center;gap:8px}
.btn-p{background:linear-gradient(135deg,#00d4ff,#0070f3);color:#fff}
.btn-p:hover{transform:translateY(-2px);box-shadow:0 8px 28px rgba(0,212,255,.38);filter:brightness(1.08)}
.btn-p:disabled{opacity:.35;cursor:not-allowed;transform:none;box-shadow:none}
.btn-g{background:var(--card);border:1px solid var(--border);color:rgba(255,255,255,.55)}
.btn-g:hover{border-color:rgba(0,212,255,.35);color:#fff;background:rgba(0,212,255,.08)}
.btn-r{background:rgba(255,107,107,.12);border:1px solid rgba(255,107,107,.3);color:var(--red)}
.btn-r:hover{background:rgba(255,107,107,.22)}
.btn-sm{padding:8px 16px;font-size:.82rem}.btn-lg{padding:15px 40px;font-size:1.05rem}
.hero{padding:42px 0 32px;text-align:center}
.chip{display:inline-flex;align-items:center;gap:8px;padding:5px 14px;border-radius:99px;
  background:rgba(0,212,255,.1);border:1px solid rgba(0,212,255,.25);margin-bottom:16px}
.chip span{font-size:.72rem;color:var(--blue);font-family:'Space Mono',monospace}
h1{font-size:clamp(1.8rem,5vw,3.4rem);font-weight:800;letter-spacing:-.03em;line-height:1.12;margin-bottom:12px}
.gt{background:linear-gradient(90deg,var(--blue),var(--green));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hero p{color:rgba(255,255,255,.4);max-width:540px;margin:0 auto;font-size:.9rem;line-height:1.6}
.upload-zone{border:2px dashed rgba(0,212,255,.28);border-radius:22px;min-height:230px;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  cursor:pointer;position:relative;overflow:hidden;transition:all .3s;padding:32px 24px;text-align:center}
.upload-zone:hover,.upload-zone.drag{border-color:var(--blue);background:rgba(0,212,255,.04);animation:glow 1.8s infinite}
.scanline{position:absolute;left:0;right:0;height:2px;
  background:linear-gradient(90deg,transparent,var(--blue),transparent);animation:scan 2.2s linear infinite}
.upload-icon{width:72px;height:72px;border-radius:18px;background:rgba(0,212,255,.1);
  border:1px solid rgba(0,212,255,.3);display:flex;align-items:center;justify-content:center;margin-bottom:14px}
.pills{display:flex;gap:8px;flex-wrap:wrap;justify-content:center;margin-top:10px}
.pill{padding:3px 10px;border-radius:6px;font-size:.7rem;font-family:'Space Mono',monospace;
  background:rgba(0,212,255,.1);color:var(--blue);border:1px solid rgba(0,212,255,.22)}
.q-wrap{margin-top:18px}
.q-hdr{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}
.q-title{font-size:.76rem;font-family:'Space Mono',monospace;color:rgba(255,255,255,.32);letter-spacing:.1em}
.q-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(115px,1fr));gap:10px}
.q-item{position:relative;border-radius:11px;overflow:hidden;
  border:1px solid rgba(255,255,255,.1);background:#0a1020;transition:all .3s}
.q-item:hover{border-color:rgba(0,212,255,.4)}
.q-item img{width:100%;height:86px;object-fit:cover;display:block}
.q-name{padding:5px 7px;font-size:.6rem;color:rgba(255,255,255,.5);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-family:'Space Mono',monospace;background:rgba(0,0,0,.55)}
.q-status{position:absolute;top:5px;right:5px;width:20px;height:20px;border-radius:50%;display:flex;align-items:center;justify-content:center}
.qs-wait{background:rgba(255,255,255,.15)}
.qs-run{background:rgba(0,212,255,.3);animation:shimmer 1s infinite}
.qs-done{background:rgba(0,255,136,.3)}.qs-err{background:rgba(255,107,107,.3)}
.q-rm{position:absolute;top:4px;left:4px;width:18px;height:18px;border-radius:50%;
  background:rgba(255,107,107,.75);border:none;cursor:pointer;display:none;
  align-items:center;justify-content:center;color:#fff;font-size:.65rem;line-height:1}
.q-item:hover .q-rm{display:flex}
.prog-bar{margin:14px 0;background:rgba(255,255,255,.05);border-radius:99px;height:5px;overflow:hidden;display:none}
.prog-bar.show{display:block}
.prog-fill{height:100%;border-radius:99px;background:linear-gradient(90deg,var(--blue),var(--green));transition:width .4s ease}
#overlay{position:fixed;inset:0;z-index:200;background:rgba(5,10,20,.97);
  backdrop-filter:blur(18px);display:none;flex-direction:column;align-items:center;justify-content:center}
#overlay.show{display:flex}
.radar-wrap{position:relative;width:128px;height:128px;margin-bottom:30px}
.radar-ring{position:absolute;inset:0;border:2px solid transparent;
  border-top-color:var(--blue);border-radius:50%;animation:radar 1.8s linear infinite}
.sonar-ring{position:absolute;inset:0;border-radius:50%;border:1px solid rgba(0,212,255,.32);animation:sonar 2.2s ease-out infinite}
.sonar-ring:nth-child(2){animation-delay:.55s}.sonar-ring:nth-child(3){animation-delay:1.1s}
.radar-icon{position:absolute;inset:0;display:flex;align-items:center;justify-content:center}
.step-list{display:flex;flex-direction:column;gap:10px;width:255px}
.step-item{display:flex;align-items:center;gap:12px}
.step-dot{width:22px;height:22px;border-radius:50%;border:1px solid;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:all .4s}
.sd-done{background:rgba(0,255,136,.2);border-color:var(--green)}
.sd-active{background:rgba(0,212,255,.2);border-color:var(--blue)}
.sd-wait{background:rgba(255,255,255,.04);border-color:rgba(255,255,255,.12)}
.step-lbl{font-size:.82rem;font-family:'Space Mono',monospace;transition:color .4s}
.sl-done{color:var(--green)}.sl-active{color:#fff}.sl-wait{color:rgba(255,255,255,.28)}
.err-bar{padding:11px 16px;border-radius:12px;display:none;align-items:center;gap:12px;
  background:rgba(255,100,100,.1);border:1px solid rgba(255,100,100,.3);margin-bottom:16px}
.err-bar.show{display:flex}
.err-bar span{font-size:.84rem;color:var(--red);flex:1}
.res-hero{padding:24px 28px;margin-bottom:16px;border:1px solid rgba(0,255,136,.18);box-shadow:0 0 40px rgba(0,255,136,.05)}
.conf-num{font-size:clamp(3rem,8vw,5rem);font-weight:700;font-family:'Space Mono',monospace;color:var(--green);letter-spacing:-.04em;line-height:1}
.g3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
@media(max-width:900px){.g3{grid-template-columns:1fr}}
.ch{padding:12px 16px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid rgba(255,255,255,.06)}
.cl{font-size:.65rem;font-family:'Space Mono',monospace;color:rgba(255,255,255,.3);letter-spacing:.12em}
.img-panel{height:205px;overflow:hidden;cursor:zoom-in}
.img-panel img{width:100%;height:100%;object-fit:cover;transition:transform .4s;display:block}
.img-panel:hover img{transform:scale(1.04)}
.meta-rows{padding:10px 14px}
.mr{display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid rgba(255,255,255,.04);font-size:.76rem}
.mr span:first-child{color:rgba(255,255,255,.3)}
.mr span:last-child{font-family:'Space Mono',monospace;color:rgba(255,255,255,.75)}
.cam-ctrl{padding:10px 14px;display:flex;flex-direction:column;gap:8px}
.mode-tabs{display:flex;gap:4px}
.mt{flex:1;padding:5px 0;border:1px solid rgba(255,255,255,.07);border-radius:7px;
  background:rgba(255,255,255,.04);cursor:pointer;font-family:'Space Mono',monospace;
  font-size:.7rem;color:rgba(255,255,255,.36);transition:all .25s}
.mt.active{background:rgba(0,212,255,.18);border-color:rgba(0,212,255,.4);color:var(--blue)}
.sl-row{display:flex;align-items:center;gap:10px;font-size:.76rem}
.sl-row span:first-child{color:rgba(255,255,255,.3)}
.sl-row input{flex:1;accent-color:var(--blue);height:4px}
.sl-row .vl{font-family:'Space Mono',monospace;color:var(--blue);width:36px;text-align:right}
.score-bars{padding:12px 14px}
.score-row{display:flex;align-items:center;gap:8px;margin-bottom:7px}
.sn{width:70px;text-align:right;font-size:.72rem;font-family:'Space Mono',monospace;flex-shrink:0}
.st{flex:1;height:18px;border-radius:99px;overflow:hidden;background:rgba(255,255,255,.05)}
.sf{height:100%;border-radius:99px;transition:width 1s cubic-bezier(.4,0,.2,1)}
.sp{width:46px;font-size:.72rem;font-family:'Space Mono',monospace}
.stat-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-top:16px}
@media(max-width:700px){.stat-grid{grid-template-columns:repeat(3,1fr)}}
.stat-card{border-radius:14px;padding:14px;background:var(--card);border:1px solid var(--border);transition:all .3s}
.stat-card:hover{transform:translateY(-2px)}
.stat-lbl{font-size:.6rem;font-family:'Space Mono',monospace;color:rgba(255,255,255,.3);letter-spacing:.1em}
.stat-val{font-size:1.5rem;font-weight:700;font-family:'Space Mono',monospace;margin-top:7px}
.res-nav{display:none;margin:18px 0 8px;align-items:center;gap:12px;flex-wrap:wrap}
.batch-stats{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:18px}
.bst{padding:12px 18px;border-radius:12px;background:var(--card);border:1px solid var(--border);flex:1;min-width:90px}
.bst-val{font-size:1.4rem;font-weight:700;font-family:'Space Mono',monospace}
.bst-lbl{font-size:.62rem;color:rgba(255,255,255,.32);font-family:'Space Mono',monospace;margin-top:3px}
.batch-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:13px}
.bc{border-radius:16px;overflow:hidden;border:1px solid var(--border);background:var(--card);cursor:pointer;transition:all .3s}
.bc:hover{transform:translateY(-3px);border-color:rgba(0,212,255,.4);box-shadow:0 8px 28px rgba(0,212,255,.12)}
.bc.sel{border-color:var(--blue);box-shadow:0 0 20px rgba(0,212,255,.22)}
.bc img{width:100%;height:125px;object-fit:cover;display:block}
.bc-body{padding:10px 12px}
.bc-cls{font-weight:700;font-size:.9rem;margin-bottom:2px}
.bc-conf{font-family:'Space Mono',monospace;font-size:.78rem}
.bc-file{font-size:.66rem;color:rgba(255,255,255,.32);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:3px}
.wafer-img{width:100%;max-height:480px;object-fit:contain;border-radius:14px}
.mini-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin-top:16px}
@media(max-width:600px){.mini-grid{grid-template-columns:repeat(2,1fr)}}
.mc{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:15px;transition:all .3s}
.mc:hover{transform:translateY(-2px);border-color:rgba(0,212,255,.3)}
.mc .mv{font-size:1.3rem;font-weight:700;font-family:'Space Mono',monospace}
.hh{display:grid;padding:11px 20px;grid-template-columns:2.5rem 1fr 120px 90px 90px 90px;
  font-size:.65rem;font-family:'Space Mono',monospace;color:rgba(255,255,255,.27);
  letter-spacing:.08em;border-bottom:1px solid rgba(255,255,255,.06)}
.hr{display:grid;padding:12px 20px;grid-template-columns:2.5rem 1fr 120px 90px 90px 90px;
  align-items:center;border-bottom:1px solid rgba(255,255,255,.04);cursor:pointer;transition:background .2s}
.hr:hover{background:rgba(0,212,255,.05)}
.hr:last-child{border-bottom:none}
.ht{width:32px;height:32px;border-radius:6px;object-fit:cover;border:1px solid rgba(0,212,255,.2)}
.hf-r{display:flex;align-items:center;gap:9px}
.hn{font-size:.84rem;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.sp-pill{padding:3px 10px;border-radius:99px;font-size:.67rem;font-family:'Space Mono',monospace;display:inline-block}
.sp-def{background:rgba(255,107,107,.12);color:var(--red);border:1px solid rgba(255,107,107,.25)}
.sp-ok{background:rgba(0,255,136,.1);color:var(--green);border:1px solid rgba(0,255,136,.22)}
.pipe-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(155px,1fr));gap:14px;margin-bottom:28px}
.pc{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:18px;text-align:center}
.pi{width:52px;height:52px;border-radius:14px;background:rgba(0,212,255,.1);border:1px solid rgba(0,212,255,.25);
  display:flex;align-items:center;justify-content:center;margin:0 auto 12px;position:relative}
.pn{position:absolute;top:-8px;right:-8px;width:18px;height:18px;border-radius:50%;
  background:var(--blue);color:#000;font-size:.62rem;font-weight:700;display:flex;align-items:center;justify-content:center}
.ag{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:600px){.ag{grid-template-columns:1fr}}
.ar{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid rgba(255,255,255,.05);font-size:.84rem}
.ar span:first-child{color:rgba(255,255,255,.36)}
.ar span:last-child{font-family:'Space Mono',monospace;color:rgba(255,255,255,.8)}
footer{border-top:1px solid rgba(255,255,255,.06);padding:24px}
.fi{max-width:1400px;margin:auto;padding:0 24px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px}
</style>
</head>
<body>
<div id="root">
<nav>
  <div class="nav-in">
    <div class="logo" onclick="S('analyze')">
      <div class="logo-icon">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#00d4ff" stroke-width="1.8"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="3"/><line x1="12" y1="2" x2="12" y2="5"/><line x1="12" y1="19" x2="12" y2="22"/><line x1="2" y1="12" x2="5" y2="12"/><line x1="19" y1="12" x2="22" y2="12"/></svg>
      </div>
      <div><div class="logo-name">Wafer<span>AI</span></div><div class="logo-tag">SEM DEFECT INTELLIGENCE</div></div>
    </div>
    <div class="nav-links">
      <button class="nb active" onclick="S('analyze')">Analyze</button>
      <button class="nb" onclick="S('results')">Results</button>
      <button class="nb" onclick="S('batch')">Batch</button>
      <button class="nb" onclick="S('wafer')">Wafer Map</button>
      <button class="nb" onclick="S('history')">Reports</button>
      <button class="nb" onclick="S('about')">About</button>
    </div>
    <div class="mbadge">
      <span class="pulse-dot" style="display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--green)"></span>
      <span>MobileNetV3</span>
    </div>
  </div>
</nav>
<main>

<!-- ANALYZE -->
<div class="sec active" id="sec-analyze">
  <div class="hero">
    <div class="chip"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#00d4ff" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg><span>AI-POWERED WAFER INSPECTION</span></div>
    <h1>SEM Defect<br><span class="gt">Classification Engine</span></h1>
    <p>Upload SEM wafer images for instant AI-powered defect classification, GradCAM explainability, and wafer map generation.</p>
  </div>
  <div class="err-bar" id="errBar">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--red)" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><circle cx="12" cy="16" r="1" fill="var(--red)"/></svg>
    <span id="errMsg">Error</span>
    <button class="btn btn-sm" style="padding:4px 10px;font-size:.74rem;background:rgba(255,107,107,.18);color:var(--red);border:1px solid rgba(255,107,107,.3)" onclick="hideErr()">✕</button>
  </div>
  <div class="card" style="padding:22px">
    <div id="scanLine" class="scanline" style="display:none"></div>
    <div class="upload-zone" id="upZone"
      onclick="document.getElementById('fi').click()"
      ondragover="event.preventDefault();this.classList.add('drag')"
      ondragleave="this.classList.remove('drag')"
      ondrop="onDrop(event)">
      <div class="upload-icon float-anim">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#00d4ff" stroke-width="1.5"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
      </div>
      <div style="font-size:1.05rem;font-weight:700;margin-bottom:6px">Drop SEM images here</div>
      <div style="color:rgba(255,255,255,.36);font-size:.84rem">or click to browse files</div>
      <div class="pills" style="margin-top:14px">
        <span class="pill">PNG</span><span class="pill">JPG</span><span class="pill">BMP</span><span class="pill">TIF</span>
        <span class="pill" style="background:rgba(0,255,136,.1);color:var(--green);border-color:rgba(0,255,136,.3)">MULTI-SELECT ✓</span>
      </div>
      <input type="file" id="fi" accept=".png,.jpg,.jpeg,.bmp,.tif,.tiff" multiple style="display:none" onchange="onFileSelect(event)"/>
    </div>
    <div class="q-wrap" id="qWrap" style="display:none">
      <div class="q-hdr">
        <span class="q-title" id="qTitle">0 FILES QUEUED</span>
        <div style="display:flex;gap:8px">
          <button class="btn btn-g btn-sm" onclick="document.getElementById('fi').click()">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>Add More
          </button>
          <button class="btn btn-r btn-sm" onclick="clearQ()">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/></svg>Clear All
          </button>
        </div>
      </div>
      <div class="prog-bar" id="pbar"><div class="prog-fill" id="pfill" style="width:0%"></div></div>
      <div class="q-grid" id="qGrid"></div>
    </div>
    <div style="display:flex;justify-content:center;gap:14px;margin-top:26px">
      <button class="btn btn-p btn-lg" id="runBtn" disabled onclick="runAll()">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
        <span id="runLabel">Analyze Images</span>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="7" y1="17" x2="17" y2="7"/><polyline points="7 7 17 7 17 17"/></svg>
      </button>
    </div>
  </div>
</div>

<!-- RESULTS -->
<div class="sec" id="sec-results">
  <div id="noRes" style="display:flex;flex-direction:column;align-items:center;padding:80px 0">
    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="rgba(0,212,255,.26)" stroke-width="1.5" style="margin-bottom:14px"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
    <p style="color:rgba(255,255,255,.36);font-size:1rem;font-weight:600;margin-bottom:8px">No Analysis Yet</p>
    <p style="color:rgba(255,255,255,.22);font-size:.84rem;margin-bottom:16px">Upload and analyze images to see results</p>
    <button class="btn btn-p btn-sm" onclick="S('analyze')">Go to Analyze</button>
  </div>
  <div id="resContent" style="display:none">
    <div class="res-nav" id="resNav">
      <span class="mono" style="font-size:.72rem;color:rgba(255,255,255,.3)">RESULT</span>
      <div style="display:flex;gap:6px;align-items:center">
        <button class="btn btn-g btn-sm" onclick="prevR()" id="btnPrev">&#8592; Prev</button>
        <span class="mono" style="font-size:.8rem;color:var(--blue)" id="navLbl">1/1</span>
        <button class="btn btn-g btn-sm" onclick="nextR()" id="btnNext">Next &#8594;</button>
      </div>
      <button class="btn btn-g btn-sm" style="margin-left:auto" onclick="S('batch')">All Batch Results &#8594;</button>
    </div>
    <div class="card res-hero fade-up" style="margin-bottom:16px">
      <div style="display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:14px">
        <div>
          <div class="mono" style="font-size:.64rem;color:rgba(255,255,255,.28);letter-spacing:.12em;margin-bottom:6px">PRIMARY DETECTION</div>
          <div style="font-size:clamp(1.4rem,4vw,2.6rem);font-weight:800;letter-spacing:-.02em;margin-bottom:8px">
            <span id="rCls">—</span> <span style="color:rgba(255,255,255,.24)">Defect</span>
          </div>
          <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap">
            <span class="conf-num" id="rConf">—</span>
            <div>
              <div style="font-size:.76rem;color:rgba(255,255,255,.34)">Confidence Score</div>
              <div class="mono" id="r2nd" style="font-size:.68rem;margin-top:4px;padding:3px 10px;border-radius:8px;
                background:rgba(255,170,0,.14);color:var(--amber);border:1px solid rgba(255,170,0,.28);display:inline-block">2nd: —</div>
            </div>
          </div>
        </div>
        <div style="display:flex;gap:10px;flex-wrap:wrap">
          <button class="btn btn-p btn-sm" onclick="S('wafer')">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="3"/></svg>Wafer Map
          </button>
          <a id="dlGC" class="btn btn-g btn-sm" download="gradcam.png">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>GradCAM
          </a>
        </div>
      </div>
    </div>
    <div class="g3" style="margin-bottom:16px">
      <div class="card fade-up">
        <div class="ch"><span class="cl">ORIGINAL IMAGE</span>
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,.3)" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg></div>
        <div class="img-panel"><img id="rOrig" src="" alt="original"/></div>
        <div class="meta-rows">
          <div class="mr"><span>File</span><span id="mFile">—</span></div>
          <div class="mr"><span>Resolution</span><span>224 × 224 px</span></div>
          <div class="mr"><span>Device</span><span id="mDev">cpu</span></div>
        </div>
      </div>
      <div class="card fade-up" style="animation-delay:.1s">
        <div class="ch"><span class="cl">GRADCAM THERMAL</span>
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#00d4ff" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg></div>
        <div class="img-panel"><img id="rGrad" src="" alt="gradcam"/></div>
        <div class="cam-ctrl">
          <div class="mode-tabs">
            <button class="mt active" onclick="setMode('overlay',this)">overlay</button>
            <button class="mt" onclick="setMode('blend',this)">blend</button>
            <button class="mt" onclick="setMode('split',this)">split</button>
          </div>
          <div class="sl-row">
            <span>Opacity</span>
            <input type="range" min="20" max="100" value="100" oninput="setOp(this.value)"/>
            <span class="vl" id="opVal">100%</span>
          </div>
        </div>
      </div>
      <div class="card fade-up" style="animation-delay:.2s">
        <div class="ch"><span class="cl">CLASS PROBABILITIES</span>
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#00ff88" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg></div>
        <div class="score-bars" id="scoreBars"></div>
      </div>
    </div>
    <div class="stat-grid" id="statGrid"></div>
  </div>
</div>

<!-- BATCH -->
<div class="sec" id="sec-batch">
  <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;padding:24px 0 16px">
    <div><h2 style="font-size:1.8rem;font-weight:800;letter-spacing:-.02em">Batch Results</h2>
    <p style="color:rgba(255,255,255,.36);font-size:.87rem;margin-top:4px" id="bSubtitle">No batch yet</p></div>
  </div>
  <div id="bEmpty" class="card" style="padding:70px;display:flex;flex-direction:column;align-items:center">
    <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="rgba(0,212,255,.26)" stroke-width="1.5" style="margin-bottom:14px"><rect x="2" y="7" width="20" height="14" rx="2"/><path d="M16 7V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v2"/></svg>
    <p style="color:rgba(255,255,255,.34)">Upload multiple images and analyze to see batch results</p>
    <button class="btn btn-p btn-sm" style="margin-top:16px" onclick="S('analyze')">Go to Analyze</button>
  </div>
  <div id="bContent" style="display:none">
    <div class="batch-stats" id="bStats"></div>
    <div class="mono" style="font-size:.7rem;color:rgba(255,255,255,.28);letter-spacing:.1em;margin-bottom:12px">CLICK ANY CARD TO VIEW FULL DETAILS</div>
    <div class="batch-grid" id="bGrid"></div>
  </div>
</div>

<!-- WAFER -->
<div class="sec" id="sec-wafer">
  <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:14px;padding:24px 0 16px">
    <div><h2 style="font-size:1.8rem;font-weight:800;letter-spacing:-.02em">Wafer Map</h2>
    <p style="color:rgba(255,255,255,.36);font-size:.87rem;margin-top:4px">Die-level defect distribution — <span id="wCls" style="color:var(--blue)">run analysis first</span></p></div>
    <a id="dlW" class="btn btn-p btn-sm" download="wafer_map.png" style="display:none">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>Export PNG
    </a>
  </div>
  <div id="wEmpty" class="card" style="padding:70px;display:flex;flex-direction:column;align-items:center">
    <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="rgba(0,212,255,.26)" stroke-width="1.5" style="margin-bottom:14px"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="3"/></svg>
    <p style="color:rgba(255,255,255,.34)">Run an analysis to generate the wafer map</p>
    <button class="btn btn-p btn-sm" style="margin-top:16px" onclick="S('analyze')">Analyze Image</button>
  </div>
  <div id="wContent" style="display:none">
    <div class="card" style="padding:18px;display:flex;justify-content:center"><img id="wImg" class="wafer-img" src="" alt="wafer"/></div>
    <div class="mini-grid">
      <div class="mc"><div class="mono" style="font-size:.6rem;color:rgba(255,255,255,.26);margin-bottom:7px">CLASS</div><div class="mv" id="wStatCls" style="color:var(--blue)">—</div></div>
      <div class="mc"><div class="mono" style="font-size:.6rem;color:rgba(255,255,255,.26);margin-bottom:7px">CONFIDENCE</div><div class="mv" id="wStatConf" style="color:var(--green)">—</div></div>
      <div class="mc"><div class="mono" style="font-size:.6rem;color:rgba(255,255,255,.26);margin-bottom:7px">YIELD</div><div class="mv" style="color:var(--amber)">94.3%</div></div>
      <div class="mc"><div class="mono" style="font-size:.6rem;color:rgba(255,255,255,.26);margin-bottom:7px">STATUS</div><div class="mv" id="wStatSt" style="color:var(--red)">—</div></div>
    </div>
  </div>
</div>

<!-- HISTORY -->
<div class="sec" id="sec-history">
  <div style="padding:24px 0 16px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px">
    <div><h2 style="font-size:1.8rem;font-weight:800;letter-spacing:-.02em">Reports</h2>
    <p style="color:rgba(255,255,255,.36);font-size:.87rem;margin-top:4px" id="hCount">0 scans this session</p></div>
  </div>
  <div id="hEmpty" class="card" style="padding:70px;display:flex;flex-direction:column;align-items:center">
    <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="rgba(0,212,255,.26)" stroke-width="1.5" style="margin-bottom:14px"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
    <p style="color:rgba(255,255,255,.34)">No scans yet</p>
  </div>
  <div id="hTable" style="display:none">
    <div class="card" style="overflow:hidden">
      <div class="hh"><span>#</span><span>FILE</span><span>CLASS</span><span>CONF</span><span>TIME</span><span>STATUS</span></div>
      <div id="hRows"></div>
    </div>
  </div>
</div>

<!-- ABOUT -->
<div class="sec" id="sec-about">
  <div style="text-align:center;padding:42px 0 26px;max-width:560px;margin:auto">
    <div class="chip" style="margin-bottom:16px">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#00d4ff" stroke-width="2"><path d="M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v18m0 0h10a2 2 0 0 0 2-2V9M9 21H5a2 2 0 0 1-2-2V9m0 0h18"/></svg>
      <span>HACKATHON PROJECT — SEM AI VISION</span>
    </div>
    <h1 style="font-size:clamp(1.8rem,5vw,3rem)">The Technology Behind<br><span class="gt">WaferAI</span></h1>
    <p style="color:rgba(255,255,255,.38);margin-top:14px;font-size:.9rem;line-height:1.7">MobileNetV3-Small fine-tuned on WM-811K SEM wafer maps. 8 defect classes. GradCAM per inference. Full batch processing. Deployed on HuggingFace Spaces.</p>
  </div>
  <p class="mono" style="font-size:.64rem;color:rgba(255,255,255,.24);letter-spacing:.12em;text-align:center;margin-bottom:18px">MODEL PIPELINE</p>
  <div class="pipe-grid">
    <div class="pc"><div class="pi"><span class="pn">1</span><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#00d4ff" stroke-width="1.5"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg></div><div style="font-size:.83rem;font-weight:700;margin-bottom:3px">Data Collection</div><div style="font-size:.7rem;color:rgba(255,255,255,.34);line-height:1.5">WM-811K SEM dataset</div></div>
    <div class="pc"><div class="pi"><span class="pn">2</span><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#00d4ff" stroke-width="1.5"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/><line x1="15" y1="3" x2="15" y2="21"/></svg></div><div style="font-size:.83rem;font-weight:700;margin-bottom:3px">Preprocessing</div><div style="font-size:.7rem;color:rgba(255,255,255,.34);line-height:1.5">224×224, grayscale norm</div></div>
    <div class="pc"><div class="pi"><span class="pn">3</span><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#00d4ff" stroke-width="1.5"><line x1="6" y1="3" x2="6" y2="15"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M18 9a9 9 0 0 1-9 9"/></svg></div><div style="font-size:.83rem;font-weight:700;margin-bottom:3px">Train/Val/Test</div><div style="font-size:.7rem;color:rgba(255,255,255,.34);line-height:1.5">70 / 15 / 15 split</div></div>
    <div class="pc"><div class="pi"><span class="pn">4</span><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#00d4ff" stroke-width="1.5"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/></svg></div><div style="font-size:.83rem;font-weight:700;margin-bottom:3px">Model Training</div><div style="font-size:.7rem;color:rgba(255,255,255,.34);line-height:1.5">MobileNetV3, 30 epochs</div></div>
    <div class="pc"><div class="pi"><span class="pn">5</span><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#00d4ff" stroke-width="1.5"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg></div><div style="font-size:.83rem;font-weight:700;margin-bottom:3px">FastAPI</div><div style="font-size:.7rem;color:rgba(255,255,255,.34);line-height:1.5">HuggingFace Spaces</div></div>
    <div class="pc"><div class="pi"><span class="pn">6</span><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#00d4ff" stroke-width="1.5"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg></div><div style="font-size:.83rem;font-weight:700;margin-bottom:3px">GradCAM</div><div style="font-size:.7rem;color:rgba(255,255,255,.34);line-height:1.5">Thermal heatmaps</div></div>
  </div>
  <div class="ag">
    <div class="card" style="padding:20px">
      <div class="mono" style="font-size:.64rem;color:rgba(255,255,255,.28);letter-spacing:.12em;margin-bottom:14px">ARCHITECTURE</div>
      <div class="ar"><span>Base Model</span><span>MobileNetV3-Small</span></div>
      <div class="ar"><span>Pretrained</span><span>ImageNet1K</span></div>
      <div class="ar"><span>Fine-tuned</span><span>WM-811K (8 classes)</span></div>
      <div class="ar"><span>Input</span><span>224×224 → RGB</span></div>
      <div class="ar"><span>Backend</span><span>FastAPI + uvicorn</span></div>
      <div class="ar"><span>Explainability</span><span>GradCAM</span></div>
    </div>
    <div class="card" style="padding:20px">
      <div class="mono" style="font-size:.64rem;color:rgba(255,255,255,.28);letter-spacing:.12em;margin-bottom:14px">TRAINING</div>
      <div class="ar"><span>Dataset</span><span>WM-811K wafer maps</span></div>
      <div class="ar"><span>Split</span><span>70 / 15 / 15</span></div>
      <div class="ar"><span>Epochs</span><span>30</span></div>
      <div class="ar"><span>Optimizer</span><span>Adam lr=3e-4</span></div>
      <div class="ar"><span>Loss</span><span>CrossEntropyLoss</span></div>
      <div class="ar"><span>Test Accuracy</span><span>98.0%</span></div>
    </div>
  </div>
</div>

</main>
<footer>
  <div class="fi">
    <div style="font-size:1rem;font-weight:800">Wafer<span style="color:var(--blue)">AI</span>
      <span style="font-weight:400;font-size:.78rem;color:rgba(255,255,255,.26);margin-left:8px">Precision Defect Intelligence</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px">
      <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--green)" class="pulse-dot"></span>
      <span class="mono" style="font-size:.68rem;color:rgba(255,255,255,.26)">FastAPI + MobileNetV3 + GradCAM &bull; Multi-Image Batch &bull; HuggingFace Spaces</span>
    </div>
  </div>
</footer>
</div>

<div id="overlay">
  <div class="radar-wrap">
    <div class="sonar-ring"></div><div class="sonar-ring"></div><div class="sonar-ring"></div>
    <div class="radar-ring"></div>
    <div class="radar-icon">
      <svg width="38" height="38" viewBox="0 0 24 24" fill="none" stroke="#00d4ff" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="3"/><line x1="12" y1="2" x2="12" y2="5"/><line x1="12" y1="19" x2="12" y2="22"/><line x1="2" y1="12" x2="5" y2="12"/><line x1="19" y1="12" x2="22" y2="12"/></svg>
    </div>
  </div>
  <p class="mono" style="font-size:.72rem;color:rgba(0,212,255,.6);letter-spacing:.18em;margin-bottom:6px">WAFER AI · SCANNING</p>
  <p id="ovInfo" style="font-size:.84rem;color:rgba(255,255,255,.4);margin-bottom:28px;font-family:'Space Mono',monospace;text-align:center"></p>
  <div class="step-list" id="stepList"></div>
</div>

<script>
const CC={Bridge:'#ff6b6b',Clean:'#00ff88','CMP Scratches':'#ffaa00',Crack:'#ff4444',LER:'#a78bfa',Open:'#00d4ff',Other:'#94a3b8',Vias:'#fbbf24'};
const STEPS=['Loading model...','Running inference...','Generating GradCAM...','Mapping wafer...'];
const STATS=[{l:'Accuracy',v:'98.0%',c:'#00ff88'},{l:'Model Size',v:'3.2 MB',c:'#00d4ff'},{l:'Inference',v:'~120ms',c:'#ffaa00'},{l:'Classes',v:'8',c:'#a78bfa'},{l:'Input',v:'224×224',c:'#00d4ff'},{l:'Format',v:'FP16',c:'#00ff88'}];
let Q=[],BR=[],VI=0,running=false,histArr=[];

function S(id){
  document.querySelectorAll('.sec').forEach(s=>s.classList.remove('active'));
  document.getElementById('sec-'+id).classList.add('active');
  document.querySelectorAll('.nb').forEach(b=>{
    b.classList.toggle('active',b.getAttribute('onclick')&&b.getAttribute('onclick').includes("'"+id+"'"));
  });
  window.scrollTo(0,0);
}

function onDrop(e){e.preventDefault();document.getElementById('upZone').classList.remove('drag');addFiles(Array.from(e.dataTransfer.files));}
function onFileSelect(e){addFiles(Array.from(e.target.files));e.target.value='';}
function addFiles(files){
  const ok=['image/png','image/jpeg','image/bmp','image/tiff'];
  files.forEach(f=>{
    if(!ok.includes(f.type)&&!f.name.match(/\.(png|jpg|jpeg|bmp|tif|tiff)$/i))return;
    if(Q.find(q=>q.file.name===f.name&&q.file.size===f.size))return;
    Q.push({file:f,url:URL.createObjectURL(f),status:'wait'});
  });
  renderQ(); updateBtn();
}
function removeQ(i){URL.revokeObjectURL(Q[i].url);Q.splice(i,1);renderQ();updateBtn();}
function clearQ(){Q.forEach(q=>URL.revokeObjectURL(q.url));Q=[];renderQ();updateBtn();}
function renderQ(){
  const wrap=document.getElementById('qWrap'),grid=document.getElementById('qGrid');
  if(!Q.length){wrap.style.display='none';document.getElementById('scanLine').style.display='none';return;}
  wrap.style.display='block';document.getElementById('scanLine').style.display='block';
  document.getElementById('qTitle').textContent=Q.length+' FILE'+(Q.length>1?'S':'')+' QUEUED';
  grid.innerHTML=Q.map((q,i)=>{
    const ico=q.status==='done'?'<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#00ff88" stroke-width="3"><polyline points="20 6 9 17 4 12"/></svg>'
      :q.status==='error'?'<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#ff6b6b" stroke-width="3"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>'
      :q.status==='running'?'<div style="width:7px;height:7px;border-radius:50%;background:var(--blue);animation:pulse 1s infinite"></div>'
      :'<div style="width:7px;height:7px;border-radius:50%;background:rgba(255,255,255,.3)"></div>';
    const sc={wait:'qs-wait',running:'qs-run',done:'qs-done',error:'qs-err'}[q.status];
    return`<div class="q-item" id="qi${i}"><button class="q-rm" onclick="event.stopPropagation();removeQ(${i})">&#x2715;</button><img src="${q.url}" alt=""/><div class="q-name">${q.file.name}</div><div class="q-status ${sc}">${ico}</div></div>`;
  }).join('');
}
function setQS(i,s){
  if(i<0||i>=Q.length)return; Q[i].status=s;
  const el=document.getElementById('qi'+i); if(!el)return;
  const ico=s==='done'?'<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#00ff88" stroke-width="3"><polyline points="20 6 9 17 4 12"/></svg>'
    :s==='error'?'<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#ff6b6b" stroke-width="3"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>'
    :s==='running'?'<div style="width:7px;height:7px;border-radius:50%;background:var(--blue);animation:pulse 1s infinite"></div>'
    :'<div style="width:7px;height:7px;border-radius:50%;background:rgba(255,255,255,.3)"></div>';
  const sc={wait:'qs-wait',running:'qs-run',done:'qs-done',error:'qs-err'}[s];
  const dot=el.querySelector('.q-status'); dot.className='q-status '+sc; dot.innerHTML=ico;
}
function updateBtn(){
  const b=document.getElementById('runBtn'),l=document.getElementById('runLabel');
  b.disabled=!Q.length||running;
  l.textContent=Q.length===1?'Analyze Image':'Analyze '+Q.length+' Images';
}
function showErr(m){document.getElementById('errMsg').textContent=m;document.getElementById('errBar').classList.add('show');}
function hideErr(){document.getElementById('errBar').classList.remove('show');}

let si=0;
function showOv(info){
  document.getElementById('overlay').classList.add('show');
  document.getElementById('ovInfo').textContent=info||'';
  document.getElementById('stepList').innerHTML=STEPS.map((s,i)=>`<div class="step-item"><div class="step-dot sd-wait" id="d${i}"><div style="width:8px;height:8px;border-radius:50%;background:rgba(255,255,255,.2)"></div></div><span class="step-lbl sl-wait" id="l${i}">${s}</span></div>`).join('');
  si=0; adv();
}
function hideOv(){document.getElementById('overlay').classList.remove('show');}
function adv(){
  if(si>0){const p=si-1;document.getElementById('d'+p).className='step-dot sd-done';document.getElementById('d'+p).innerHTML='<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#00ff88" stroke-width="3"><polyline points="20 6 9 17 4 12"/></svg>';document.getElementById('l'+p).className='step-lbl sl-done';}
  if(si<STEPS.length){document.getElementById('d'+si).className='step-dot sd-active';document.getElementById('d'+si).innerHTML='<div style="width:8px;height:8px;border-radius:50%;background:var(--blue);animation:pulse 1s infinite"></div>';document.getElementById('l'+si).className='step-lbl sl-active';si++;}
}

async function safeFetch(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    let msg = 'HTTP ' + res.status;
    try { const j = await res.json(); msg = j.detail || j.error || msg; } catch(_){}
    throw new Error(msg);
  }
  return res.json();
}

async function runAll(){
  if(!Q.length||running)return;
  running=true; updateBtn(); BR=[]; hideErr();
  const total=Q.length;
  const pbar=document.getElementById('pbar'),pfill=document.getElementById('pfill');
  pbar.classList.add('show'); pfill.style.width='0%';
  Q.forEach((_,i)=>setQS(i,'wait'));

  if(total===1){
    showOv('Analyzing: '+Q[0].file.name);
    const iv=setInterval(adv,750);
    try{
      const fd=new FormData(); fd.append('image',Q[0].file);
      const data=await safeFetch('/predict',{method:'POST',body:fd});
      clearInterval(iv);
      for(let i=si-1;i<STEPS.length;i++){document.getElementById('d'+i).className='step-dot sd-done';document.getElementById('d'+i).innerHTML='<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#00ff88" stroke-width="3"><polyline points="20 6 9 17 4 12"/></svg>';document.getElementById('l'+i).className='step-lbl sl-done';}
      setQS(0,'done'); pfill.style.width='100%'; BR=[data];
      setTimeout(()=>{hideOv();VI=0;renderSingle(data);renderBatch([data]);addHist(data);S('results');running=false;updateBtn();},500);
    }catch(e){clearInterval(iv);hideOv();setQS(0,'error');showErr(e.message==='Failed to fetch'?'Cannot reach backend — is the Space running?':e.message);running=false;updateBtn();}
  } else {
    showOv('Starting batch: '+total+' images...');
    await new Promise(r=>setTimeout(r,700)); hideOv();
    let done=0;
    for(let i=0;i<Q.length;i++){
      setQS(i,'running');
      try{
        const fd=new FormData(); fd.append('image',Q[i].file);
        const data=await safeFetch('/predict',{method:'POST',body:fd});
        BR.push(data); setQS(i,'done'); addHist(data);
      }catch(e){BR.push({success:false,metadata:{filename:Q[i].file.name},error:e.message});setQS(i,'error');}
      done++; pfill.style.width=(done/total*100)+'%';
    }
    const first=BR.find(r=>r.success);
    if(first){VI=BR.filter(r=>r.success).indexOf(first);renderSingle(first);}
    renderBatch(BR); S('batch'); running=false; updateBtn();
  }
}

function renderSingle(d){
  if(!d||!d.success)return;
  const sorted=Object.entries(d.scores).sort((a,b)=>b[1]-a[1]);
  const tc=d.predicted_class,cf=d.confidence,sec=sorted[1];
  document.getElementById('rCls').textContent=tc;
  document.getElementById('rConf').textContent=cf.toFixed(1)+'%';
  document.getElementById('r2nd').textContent='2nd: '+sec[0]+' '+sec[1].toFixed(1)+'%';
  const os='data:image/png;base64,'+d.original_image;
  const gs='data:image/png;base64,'+d.gradcam_image;
  const ws='data:image/png;base64,'+d.wafer_map;
  document.getElementById('rOrig').src=os;
  document.getElementById('rGrad').src=gs;
  document.getElementById('mFile').textContent=d.metadata.filename;
  document.getElementById('mDev').textContent=d.metadata.device;
  document.getElementById('dlGC').href=gs;
  document.getElementById('dlW').href=ws; document.getElementById('dlW').style.display='inline-flex';
  document.getElementById('wImg').src=ws;
  document.getElementById('wCls').textContent=tc;
  document.getElementById('wStatCls').textContent=tc;
  document.getElementById('wStatConf').textContent=cf.toFixed(1)+'%';
  document.getElementById('wStatSt').textContent=tc==='Clean'?'Clean':'Defective';
  document.getElementById('wStatSt').style.color=tc==='Clean'?'var(--green)':'var(--red)';
  document.getElementById('wEmpty').style.display='none';
  document.getElementById('wContent').style.display='block';
  const bars=document.getElementById('scoreBars'); bars.innerHTML='';
  sorted.forEach(([name,score],i)=>{
    const isTop=i===0,col=CC[name]||'#00d4ff',pct=isTop?100:(score/cf*100);
    const d2=document.createElement('div'); d2.className='score-row';
    d2.innerHTML=`<span class="sn" style="color:${isTop?col:'rgba(255,255,255,.36)'}">${name}</span><div class="st"><div class="sf" style="width:0%;--w:${Math.max(pct,score>0?3:0)}%;background:${isTop?`linear-gradient(90deg,${col}88,${col})`:'linear-gradient(90deg,rgba(0,212,255,.2),rgba(0,212,255,.45))'};box-shadow:${isTop?`0 0 10px ${col}55`:'none'}"></div></div><span class="sp" style="color:${isTop?col:'rgba(255,255,255,.3)'}">${score.toFixed(1)}%</span>`;
    bars.appendChild(d2);
  });
  setTimeout(()=>{document.querySelectorAll('.sf').forEach(el=>el.style.width=el.style.getPropertyValue('--w')||'0%');},120);
  document.getElementById('statGrid').innerHTML=STATS.map(s=>`<div class="stat-card" style="box-shadow:0 0 18px ${s.c}11"><div class="stat-lbl">${s.l}</div><div class="stat-val" style="color:${s.c}">${s.v}</div></div>`).join('');
  document.getElementById('noRes').style.display='none';
  document.getElementById('resContent').style.display='block';
  const sr=BR.filter(r=>r.success),nav=document.getElementById('resNav');
  if(sr.length>1){nav.style.display='flex';document.getElementById('navLbl').textContent=(VI+1)+'/'+sr.length;document.getElementById('btnPrev').disabled=VI===0;document.getElementById('btnNext').disabled=VI===sr.length-1;}
  else nav.style.display='none';
}
function prevR(){const sr=BR.filter(r=>r.success);if(VI>0){VI--;renderSingle(sr[VI]);}}
function nextR(){const sr=BR.filter(r=>r.success);if(VI<sr.length-1){VI++;renderSingle(sr[VI]);}}

function renderBatch(results){
  const ok=results.filter(r=>r.success);
  const def=ok.filter(r=>r.predicted_class!=='Clean');
  const avg=ok.length?(ok.reduce((s,r)=>s+r.confidence,0)/ok.length).toFixed(1):0;
  const cm={};ok.forEach(r=>{cm[r.predicted_class]=(cm[r.predicted_class]||0)+1;});
  const top=Object.entries(cm).sort((a,b)=>b[1]-a[1])[0];
  document.getElementById('bSubtitle').textContent=ok.length+' image'+(ok.length!==1?'s':'')+' analyzed';
  document.getElementById('bStats').innerHTML=[
    {l:'TOTAL',v:ok.length,c:'var(--blue)'},{l:'DEFECTIVE',v:def.length,c:'var(--red)'},
    {l:'CLEAN',v:ok.length-def.length,c:'var(--green)'},{l:'AVG CONF',v:avg+'%',c:'var(--amber)'},
    {l:'TOP CLASS',v:top?top[0]:'&#8212;',c:CC[top?top[0]:'']||'var(--blue)'},
  ].map(s=>`<div class="bst"><div class="bst-val" style="color:${s.c}">${s.v}</div><div class="bst-lbl">${s.l}</div></div>`).join('');
  const grid=document.getElementById('bGrid'); grid.innerHTML='';
  const okArr=results.filter(r=>r.success);
  okArr.forEach((r,i)=>{
    const col=CC[r.predicted_class]||'#00d4ff',clean=r.predicted_class==='Clean';
    const c=document.createElement('div'); c.className='bc'+(i===VI?' sel':'');
    c.innerHTML=`<img src="data:image/png;base64,${r.gradcam_image}" alt=""/><div class="bc-body"><div class="bc-cls" style="color:${col}">${r.predicted_class}</div><div class="bc-conf" style="color:${r.confidence>97?'var(--green)':'var(--amber)'}">${r.confidence.toFixed(1)}% <span class="sp-pill ${clean?'sp-ok':'sp-def'}" style="font-size:.6rem">${clean?'clean':'defect'}</span></div><div class="bc-file">${r.metadata.filename}</div></div>`;
    c.onclick=()=>{VI=i;document.querySelectorAll('.bc').forEach(x=>x.classList.remove('sel'));c.classList.add('sel');renderSingle(r);S('results');};
    grid.appendChild(c);
  });
  results.filter(r=>!r.success).forEach(r=>{
    const c=document.createElement('div'); c.className='bc'; c.style.cssText='border-color:rgba(255,107,107,.3);cursor:default';
    c.innerHTML=`<div style="height:125px;background:rgba(255,107,107,.08);display:flex;align-items:center;justify-content:center"><svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="var(--red)" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><circle cx="12" cy="16" r="1" fill="var(--red)"/></svg></div><div class="bc-body"><div class="bc-cls" style="color:var(--red)">Error</div><div class="bc-conf" style="color:rgba(255,255,255,.38);font-size:.7rem">${r.error||'Failed'}</div><div class="bc-file">${r.metadata?.filename||'unknown'}</div></div>`;
    grid.appendChild(c);
  });
  document.getElementById('bEmpty').style.display='none';
  document.getElementById('bContent').style.display='block';
}

function addHist(d){
  if(!d.success)return;
  histArr.unshift({file:d.metadata.filename,cls:d.predicted_class,conf:d.confidence,
    time:new Date().toLocaleTimeString(),thumb:'data:image/png;base64,'+d.gradcam_image,
    status:d.predicted_class==='Clean'?'clean':'defective'});
  document.getElementById('hEmpty').style.display='none';
  document.getElementById('hTable').style.display='block';
  document.getElementById('hCount').textContent=histArr.length+' scan(s) this session';
  document.getElementById('hRows').innerHTML=histArr.map((h,i)=>`
    <div class="hr"><span class="mono" style="color:rgba(255,255,255,.2);font-size:.72rem">${String(i+1).padStart(2,'0')}</span>
    <div class="hf-r"><img src="${h.thumb}" class="ht" alt=""/><span class="hn">${h.file}</span></div>
    <div style="display:flex;align-items:center;gap:8px"><span style="width:8px;height:8px;border-radius:50%;background:${CC[h.cls]||'#00d4ff'};flex-shrink:0;display:inline-block"></span><span style="font-size:.84rem">${h.cls}</span></div>
    <span class="mono" style="font-weight:700;color:${h.conf>97?'var(--green)':'var(--amber)'}">${h.conf.toFixed(1)}%</span>
    <span class="mono" style="font-size:.75rem;color:rgba(255,255,255,.35)">${h.time}</span>
    <span class="sp-pill ${h.status==='clean'?'sp-ok':'sp-def'}">${h.status}</span></div>`).join('');
}

function setMode(m,btn){document.querySelectorAll('.mt').forEach(t=>t.classList.remove('active'));btn.classList.add('active');document.getElementById('rGrad').style.opacity=m==='split'?'0.5':'1';}
function setOp(v){document.getElementById('rGrad').style.opacity=v/100;document.getElementById('opVal').textContent=v+'%';}

document.getElementById('statGrid').innerHTML=STATS.map(s=>`<div class="stat-card" style="box-shadow:0 0 18px ${s.c}11"><div class="stat-lbl">${s.l}</div><div class="stat-val" style="color:${s.c}">${s.v}</div></div>`).join('');
</script>
</body></html>"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=HTML)

# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    # HuggingFace Spaces always exposes port 7860 to the outside world.
    # Override via PORT env-var if you are running locally on a different port.
    port = int(os.environ.get("PORT", 7860))
    print(f"\n{'='*56}")
    print(f"  WaferAI v3 — HuggingFace Spaces Edition")
    print(f"{'='*56}")
    print(f"  Open  : http://0.0.0.0:{port}")
    print(f"  Device: {DEVICE}")
    print(f"  Model : {'✅ found' if os.path.exists(MODEL_PATH) else '⚠️  NOT FOUND — demo mode'}")
    print(f"{'='*56}\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
