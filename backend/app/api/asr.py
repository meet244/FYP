from fastapi import APIRouter

from app.asr.catalog import METHODS, model_catalog, resolve_options

router = APIRouter(prefix="/asr", tags=["asr"])


@router.get("/models")
def models():
    return {"default": resolve_options(), "models": model_catalog(), "methods": METHODS}
