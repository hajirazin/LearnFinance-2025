"""Training and inference pipeline utilities."""

from brain_api.routes.pipelines.inference import (
    InferenceContext,
    InferenceOutcome,
    compute_data_window,
    load_model_with_fallback,
    log_inference_summary,
    sort_predictions_by_return as inference_sort_predictions,
)
from brain_api.routes.pipelines.training import (
    TrainingContext,
    TrainingMetrics,
    TrainingOutcome,
    TrainingPipeline,
    check_idempotent_version,
    execute_promotion,
    get_prior_version_info,
)
from brain_api.routes.pipelines.utils import (
    get_as_of_date,
    sort_predictions_by_return,
)

__all__ = [
    # Utils
    "get_as_of_date",
    "sort_predictions_by_return",
    # Training pipeline
    "TrainingContext",
    "TrainingMetrics",
    "TrainingOutcome",
    "TrainingPipeline",
    "check_idempotent_version",
    "execute_promotion",
    "get_prior_version_info",
    # Inference pipeline
    "InferenceContext",
    "InferenceOutcome",
    "compute_data_window",
    "load_model_with_fallback",
    "log_inference_summary",
    "inference_sort_predictions",
]
