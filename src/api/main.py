"""ChefAI - FastAPI serving (predict + allergy check + Prometheus)."""

from __future__ import annotations

import asyncio
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from src.models.config import IDX_TO_CLASS, ModelConfig
from src.models.dataset import get_eval_transforms
from src.models.model import build_model

# -------------------------------------------------------------------
# Alerjen haritasi (README'deki modul)
# -------------------------------------------------------------------
ALLERGENS: dict[str, list[str]] = {
    "gluten": [
        "manti",
        "pizza",
        "borek",
        "gozleme",
        "kisir",
        "bulgur_pilavi",
        "kek",
        "kurabiye",
        "pankek",
        "avokado_tost",
        "lahmacun",  # YENI
        "simit",  # YENI
        "baklava",  # YENI
    ],
    "sut": [
        "sutlac",
        "kek",
        "kurabiye",
        "pankek",
        "borek",
        "kofte",
        "omlet",
        "baklava",  # YENI (tereyagi)
    ],
    "yumurta": [
        "omlet",
        "kek",
        "kurabiye",
        "pankek",
        "kofte",
        "manti",
        "borek",
        "baklava",  # YENI (bazi tariflerde)
    ],
    "et": [
        "kofte",
        "karniyarik",
        "etli_nohut",
        "manti",
        "borek",
        "gozleme",
        "lahmacun",  # YENI (kıyma)
        "yaprak_sarma",  # YENI (genelde etli)
    ],
    "tavuk": ["izgara_tavuk", "tavuk_suyu_corba"],
    "avokado": ["avokado_tost"],
    "nohut": ["etli_nohut"],
    "balik": ["hamsi_tava"],  # YENI kategori
    "findik": ["baklava"],  # YENI kategori (fistik/findik)
}

# -------------------------------------------------------------------
# Global state
# -------------------------------------------------------------------
MODEL: torch.nn.Module | None = None
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pt")
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORM = get_eval_transforms(IMAGE_SIZE)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL
    cfg = ModelConfig()
    model = build_model(
        model_name=cfg.model_name,
        num_classes=cfg.num_classes,
        pretrained=False,
        dropout=cfg.dropout,
    )
    path = Path(MODEL_PATH)
    if path.exists():
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"[chefai-api] Model yuklendi: {path} (val_acc={ckpt.get('val_acc', '?')})")
    else:
        print(f"[chefai-api] UYARI: {path} yok, rastgele agirliklar kullaniliyor")
    MODEL = model.to(DEVICE).eval()
    yield
    MODEL = None


app = FastAPI(title="ChefAI", version="1.0.0", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)


# -------------------------------------------------------------------
# Schemas
# -------------------------------------------------------------------
class Prediction(BaseModel):
    class_name: str
    confidence: float
    top5: list[dict]


class AllergyCheck(BaseModel):
    class_name: str
    confidence: float
    allergens: list[str]
    warnings: list[str]
    safe: bool


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
async def _load_image(file: UploadFile) -> Image.Image:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(400, "Gecersiz dosya turu (image/* bekleniyor)")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Bos dosya")
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Gorsel acilamadi: {e}") from e


@torch.no_grad()
def _predict(img: Image.Image) -> Prediction:
    if MODEL is None:
        raise HTTPException(503, "Model yuklu degil")

    if not isinstance(img, Image.Image):
        raise HTTPException(
            400,
            f"Beklenen PIL Image, alinan: {type(img).__name__}",
        )

    transformed = TRANSFORM(img)
    if not isinstance(transformed, torch.Tensor):
        raise HTTPException(
            500,
            f"TRANSFORM tensor dondurmedi: {type(transformed).__name__}",
        )
    if transformed.dim() != 3:
        raise HTTPException(
            500,
            f"Beklenen (C,H,W), alinan: {tuple(transformed.shape)}",
        )

    tensor = transformed.unsqueeze(0).to(DEVICE)
    probs = MODEL(tensor).softmax(dim=1)[0]
    top5_probs, top5_idx = probs.topk(5)
    top5 = [
        {"class_name": IDX_TO_CLASS[int(i)], "confidence": float(p)}
        for p, i in zip(top5_probs, top5_idx, strict=False)
    ]
    return Prediction(class_name=top5[0]["class_name"], confidence=top5[0]["confidence"], top5=top5)


# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE),
        "model_path": MODEL_PATH,
    }


@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...)):
    img = await _load_image(file)
    # CPU/GPU-bound islemi event loop'u bloklamadan thread'e devret
    return await asyncio.to_thread(_predict, img)


@app.post("/check-allergy", response_model=AllergyCheck)
async def check_allergy(file: UploadFile = File(...), allergens: str = ""):
    """allergens: 'gluten,sut' gibi virgulle ayrilmis liste"""
    img = await _load_image(file)
    pred = await asyncio.to_thread(_predict, img)
    user_allergens = [a.strip().lower() for a in allergens.split(",") if a.strip()]
    warnings = [
        f"'{pred.class_name}' {a} icerebilir"
        for a in user_allergens
        if a in ALLERGENS and pred.class_name in ALLERGENS[a]
    ]
    return AllergyCheck(
        class_name=pred.class_name,
        confidence=pred.confidence,
        allergens=user_allergens,
        warnings=warnings,
        safe=len(warnings) == 0,
    )
