"""SAC training endpoints (full retrain + finetune).

Sub-package shape (Phase A of the always-promote plan):

* :mod:`._shared` -- constants and the request schema shared by both endpoints
* :mod:`.full` -- ``POST /train/sac/full`` and its background task
* :mod:`.finetune` -- ``POST /train/sac/finetune`` and its background task

The aggregate ``router`` here is what ``routes/training/__init__.py``
includes, so external callers see one ``/sac/*`` namespace exactly as
before the split.
"""

from fastapi import APIRouter

from .finetune import router as _finetune_router
from .full import router as _full_router
from .preflight import router as _preflight_router

router = APIRouter()
router.include_router(_full_router)
router.include_router(_finetune_router)
router.include_router(_preflight_router)

__all__ = ["router"]
