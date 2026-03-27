---
title: WaferAI
emoji: 🔬
colorFrom: blue
colorTo: cyan
sdk: docker
pinned: false
app_port: 7860
---

# WaferAI — SEM Defect Intelligence

MobileNetV3-Small wafer defect classifier with GradCAM explainability.  
**8 classes:** Bridge · Clean · CMP Scratches · Crack · LER · Open · Other · Vias

### Stack
- **Backend:** FastAPI + uvicorn (Python 3.11)
- **Model:** MobileNetV3-Small fine-tuned on WM-811K
- **Explainability:** GradCAM thermal heatmaps
- **Visualisation:** Synthetic wafer maps (matplotlib)
