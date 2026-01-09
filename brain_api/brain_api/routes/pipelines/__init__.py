"""Training and inference pipeline utilities."""

from brain_api.routes.pipelines.inference import (
    InferenceContext,
    InferenceOutcome,
    compute_data_window,
    load_model_with_fallback,
    log_inference_summary,
)
from brain_api.routes.pipelines.inference import (
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
    # Inference pipeline
    "InferenceContext",
    "InferenceOutcome",
    # Training pipeline
    "TrainingContext",
    "TrainingMetrics",
    "TrainingOutcome",
    "TrainingPipeline",
    "check_idempotent_version",
    "compute_data_window",
    "execute_promotion",
    # Utils
    "get_as_of_date",
    "get_prior_version_info",
    "inference_sort_predictions",
    "load_model_with_fallback",
    "log_inference_summary",
    "sort_predictions_by_return",
]
