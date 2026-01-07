"""Generic training pipeline for model orchestration.

This module provides a base training workflow that can be customized
for different model types (LSTM, PatchTST, etc.) while keeping the
common orchestration logic DRY.
"""

import logging
import time
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Generic, Protocol, TypeVar

logger = logging.getLogger(__name__)

# Type variables for generic typing
ConfigT = TypeVar("ConfigT")
ModelT = TypeVar("ModelT")
DatasetT = TypeVar("DatasetT")
ResultT = TypeVar("ResultT")


class ModelStorage(Protocol):
    """Protocol for model storage backends."""

    def version_exists(self, version: str) -> bool: ...
    def read_metadata(self, version: str) -> dict[str, Any] | None: ...
    def read_current_version(self) -> str | None: ...
    def write_artifacts(
        self, version: str, model: Any, feature_scaler: Any, config: Any, metadata: dict
    ) -> None: ...
    def promote_version(self, version: str) -> None: ...


@dataclass
class TrainingContext(Generic[ConfigT]):
    """Context for a training run."""

    model_name: str  # e.g., "LSTM", "PatchTST"
    symbols: list[str]
    config: ConfigT
    start_date: date
    end_date: date
    version: str


@dataclass
class TrainingMetrics:
    """Common metrics from model training."""

    train_loss: float
    val_loss: float
    baseline_loss: float


@dataclass
class TrainingOutcome:
    """Result of training pipeline execution."""

    version: str
    data_window_start: str
    data_window_end: str
    metrics: dict[str, float]
    promoted: bool
    prior_version: str | None
    from_cache: bool  # True if version already existed


def log_timing(model_name: str, step: str, duration: float) -> None:
    """Log timing information for a training step."""
    logger.info(f"[{model_name}] {step} completed in {duration:.1f}s")


def check_idempotent_version(
    storage: ModelStorage,
    version: str,
    model_name: str,
) -> dict[str, Any] | None:
    """Check if version already exists (idempotent training).
    
    Returns existing metadata if version exists, None otherwise.
    """
    if storage.version_exists(version):
        logger.info(f"[{model_name}] Version {version} already exists (idempotent)")
        return storage.read_metadata(version)
    return None


def get_prior_version_info(
    storage: ModelStorage,
    model_name: str,
) -> tuple[str | None, float | None]:
    """Get prior version and its validation loss.
    
    Returns:
        Tuple of (prior_version, prior_val_loss)
    """
    prior_version = storage.read_current_version()
    prior_val_loss = None
    
    if prior_version:
        logger.info(f"[{model_name}] Prior version: {prior_version}")
        prior_metadata = storage.read_metadata(prior_version)
        if prior_metadata:
            prior_val_loss = prior_metadata["metrics"].get("val_loss")
            logger.info(f"[{model_name}] Prior val_loss: {prior_val_loss}")
    else:
        logger.info(f"[{model_name}] No prior version exists (first model)")
    
    return prior_version, prior_val_loss


def execute_promotion(
    storage: ModelStorage,
    version: str,
    promoted: bool,
    prior_version: str | None,
    model_name: str,
) -> None:
    """Handle model promotion logic.
    
    Promotes if:
    - Model passed evaluation (promoted=True)
    - OR this is the first model (so inference has something)
    """
    if promoted or prior_version is None:
        storage.promote_version(version)
        reason = "passed evaluation" if promoted else "first model"
        logger.info(f"[{model_name}] Version {version} promoted to current ({reason})")


class TrainingPipeline(Generic[ConfigT, DatasetT, ModelT]):
    """Generic training pipeline for ML models.
    
    Subclasses should implement the abstract methods to customize
    data loading, dataset building, and model training.
    """

    def __init__(
        self,
        model_name: str,
        storage: ModelStorage,
        compute_version_fn: Callable[[date, date, list[str], ConfigT], str],
        evaluate_promotion_fn: Callable[[float, float, float | None], bool],
        create_metadata_fn: Callable[..., dict[str, Any]],
    ):
        """Initialize the training pipeline.
        
        Args:
            model_name: Name for logging (e.g., "LSTM", "PatchTST")
            storage: Model storage backend
            compute_version_fn: Function to compute deterministic version
            evaluate_promotion_fn: Function to decide on model promotion
            create_metadata_fn: Function to create training metadata
        """
        self.model_name = model_name
        self.storage = storage
        self.compute_version = compute_version_fn
        self.evaluate_promotion = evaluate_promotion_fn
        self.create_metadata = create_metadata_fn

    def run(
        self,
        symbols: list[str],
        config: ConfigT,
        start_date: date,
        end_date: date,
        load_data_fn: Callable[[], dict[str, Any]],
        build_dataset_fn: Callable[[dict[str, Any]], DatasetT],
        train_fn: Callable[[DatasetT, ConfigT], tuple[ModelT, Any, TrainingMetrics]],
    ) -> TrainingOutcome:
        """Execute the full training pipeline.
        
        Args:
            symbols: List of symbols to train on
            config: Model configuration
            start_date: Start of training data window
            end_date: End of training data window
            load_data_fn: Function that returns dict of loaded data
            build_dataset_fn: Function that builds dataset from loaded data
            train_fn: Function that trains model and returns (model, scaler, metrics)
            
        Returns:
            TrainingOutcome with version, metrics, and promotion status
        """
        logger.info(f"[{self.model_name}] Starting training for {len(symbols)} symbols")
        logger.info(f"[{self.model_name}] Data window: {start_date} to {end_date}")

        # Compute deterministic version
        version = self.compute_version(start_date, end_date, symbols, config)
        logger.info(f"[{self.model_name}] Computed version: {version}")

        # Check idempotency
        existing_metadata = check_idempotent_version(
            self.storage, version, self.model_name
        )
        if existing_metadata:
            return TrainingOutcome(
                version=version,
                data_window_start=existing_metadata["data_window"]["start"],
                data_window_end=existing_metadata["data_window"]["end"],
                metrics=existing_metadata["metrics"],
                promoted=existing_metadata["promoted"],
                prior_version=existing_metadata.get("prior_version"),
                from_cache=True,
            )

        # Load data
        logger.info(f"[{self.model_name}] Loading data...")
        t0 = time.time()
        data = load_data_fn()
        log_timing(self.model_name, "Data loading", time.time() - t0)

        # Build dataset
        logger.info(f"[{self.model_name}] Building dataset...")
        t0 = time.time()
        dataset = build_dataset_fn(data)
        log_timing(self.model_name, "Dataset building", time.time() - t0)

        # Train model
        logger.info(f"[{self.model_name}] Starting model training...")
        t0 = time.time()
        model, feature_scaler, metrics = train_fn(dataset, config)
        log_timing(self.model_name, "Training", time.time() - t0)
        logger.info(
            f"[{self.model_name}] Metrics: train_loss={metrics.train_loss:.6f}, "
            f"val_loss={metrics.val_loss:.6f}, baseline={metrics.baseline_loss:.6f}"
        )

        # Get prior version info
        prior_version, prior_val_loss = get_prior_version_info(
            self.storage, self.model_name
        )

        # Decide on promotion
        promoted = self.evaluate_promotion(
            metrics.val_loss, metrics.baseline_loss, prior_val_loss
        )
        logger.info(
            f"[{self.model_name}] Promotion decision: "
            f"{'PROMOTED' if promoted else 'NOT promoted'}"
        )

        # Create and write metadata
        metadata = self.create_metadata(
            version=version,
            data_window_start=start_date.isoformat(),
            data_window_end=end_date.isoformat(),
            symbols=symbols,
            config=config,
            train_loss=metrics.train_loss,
            val_loss=metrics.val_loss,
            baseline_loss=metrics.baseline_loss,
            promoted=promoted,
            prior_version=prior_version,
        )

        # Write artifacts
        logger.info(f"[{self.model_name}] Writing artifacts for version {version}...")
        self.storage.write_artifacts(
            version=version,
            model=model,
            feature_scaler=feature_scaler,
            config=config,
            metadata=metadata,
        )
        logger.info(f"[{self.model_name}] Artifacts written successfully")

        # Handle promotion
        execute_promotion(
            self.storage, version, promoted, prior_version, self.model_name
        )

        return TrainingOutcome(
            version=version,
            data_window_start=start_date.isoformat(),
            data_window_end=end_date.isoformat(),
            metrics={
                "train_loss": metrics.train_loss,
                "val_loss": metrics.val_loss,
                "baseline_loss": metrics.baseline_loss,
            },
            promoted=promoted,
            prior_version=prior_version,
            from_cache=False,
        )

