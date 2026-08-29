from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "likeyellow/klue-review-star-4class"
MAX_LEN = 256

# 라벨 순서 [부정, 긍정, 중립, 리뷰아님]
ANCHOR = np.array([1.0, 5.0, 3.0])          # 앞 3개에만 대응
LABELS = ["negative", "positive", "neutral", "not_a_review"]

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작할 때 모델을 한 번만 메모리에 올린다."""
    state["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_DIR)
    state["model"] = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, low_cpu_mem_usage=True
    ).eval()
    print("모델 로드 완료")
    yield
    state.clear()


app = FastAPI(title="리뷰 별점 산출 API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://likeyellow.github.io",
    ],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class ReviewIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)


class ReviewOut(BaseModel):
    star: float | None          # 리뷰가 아니면 None
    confidence: float | None
    label: str
    is_review: bool
    probs: dict


def score_text(text: str) -> ReviewOut:
    tok, model = state["tokenizer"], state["model"]
    enc = tok(text, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        p = torch.softmax(model(**enc).logits, dim=-1)[0].numpy()

    probs = {k: round(float(v), 4) for k, v in zip(LABELS, p)}
    top = int(p.argmax())

    # 리뷰가 아니면 별점을 산출하지 않는다
    if top == 3:
        return ReviewOut(star=None, confidence=None, label=LABELS[3],
                         is_review=False, probs=probs)

    # 앞 3개만 정규화해서 기댓값과 엔트로피를 구한다
    q = p[:3] / p[:3].sum()
    star = float((q * ANCHOR).sum())
    ent = float(-(q * np.log(q + 1e-9)).sum())
    conf = 1 - ent / np.log(3)

    return ReviewOut(star=round(star, 2), confidence=round(conf, 3),
                     label=LABELS[top], is_review=True, probs=probs)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state, "classes": 4}


@app.post("/review/score", response_model=ReviewOut)
def score(req: ReviewIn):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="리뷰 본문이 비어 있습니다.")
    return score_text(text)


@app.post("/review/score/batch")
def score_batch(reqs: list[ReviewIn]):
    if len(reqs) > 100:
        raise HTTPException(status_code=400, detail="한 번에 최대 100건까지 처리합니다.")
    return [score_text(r.text.strip()) for r in reqs]