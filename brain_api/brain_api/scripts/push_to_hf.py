#!/usr/bin/env python
"""CLI script to push existing model/dataset artifacts to HuggingFace Hub.

Usage:
    # Push model
    python -m brain_api.scripts.push_to_hf model --version v2025-12-30-96849e1ed625

    # Push model and make it current
    python -m brain_api.scripts.push_to_hf model --version v2025-12-30-96849e1ed625 --make-current

    # List available model versions locally
    python -m brain_api.scripts.push_to_hf list-models

Authentication:
    The huggingface_hub library automatically uses credentials from (in order):
    1. HF_TOKEN environment variable
    2. Cached token from `huggingface-cli login` (~/.cache/huggingface/token)

    Recommended: Run `huggingface-cli login` once, then no HF_TOKEN needed.

Environment Variables:
    HF_LSTM_MODEL_REPO: Target LSTM model repository (e.g., 'username/learnfinance-lstm')
    HF_TOKEN: (Optional) HuggingFace API token - not needed if logged in via CLI
"""

import argparse
import sys


def push_model(version: str, make_current: bool = False) -> int:
    """Push a local model version to HuggingFace Hub.

    Args:
        version: Model version to push (e.g., 'v2025-12-30-96849e1ed625')
        make_current: If True, also update 'main' branch to point to this version

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    from brain_api.core.config import get_hf_lstm_halal_new_model_repo
    from brain_api.storage.huggingface import HuggingFaceModelStorage
    from brain_api.storage.local import LSTMHalalNewModelStorage

    # Check HF repo is configured
    hf_repo = get_hf_lstm_halal_new_model_repo()
    if not hf_repo:
        print("Error: HF_LSTM_HALAL_NEW_MODEL_REPO environment variable not set")
        return 1

    # Load local storage
    local_storage = LSTMHalalNewModelStorage()

    # Check version exists locally
    if not local_storage.version_exists(version):
        print(f"Error: Model version '{version}' not found locally")
        print(f"       Expected at: data/models/lstm_halal_new/{version}/")
        return 1

    print(f"Loading model version: {version}")

    # Load artifacts from local storage
    try:
        config = local_storage.load_config(version)
        feature_scaler = local_storage.load_feature_scaler(version)
        model = local_storage.load_model(version, config)
        metadata = local_storage.read_metadata(version)

        if metadata is None:
            print(f"Warning: No metadata found for {version}, using minimal metadata")
            metadata = {"version": version}
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        return 1

    # Initialize HF storage and upload
    try:
        hf_storage = HuggingFaceModelStorage(repo_id=hf_repo, local_cache=local_storage)
        print(f"Uploading to HuggingFace: {hf_repo}")

        hf_info = hf_storage.upload_model(
            version=version,
            model=model,
            feature_scaler=feature_scaler,
            config=config,
            metadata=metadata,
            make_current=make_current,
        )

        print("✓ Model uploaded successfully!")
        print(f"  Repo: {hf_info.repo_id}")
        print(f"  Version: {hf_info.version}")
        print(f"  URL: https://huggingface.co/{hf_info.repo_id}/tree/{version}")

        if make_current:
            print("  ✓ Set as current (main branch)")

        return 0

    except Exception as e:
        print(f"Error uploading to HuggingFace: {e}")
        return 1


def list_local_models() -> int:
    """List available model versions in local storage.

    Returns:
        Exit code (0 for success)
    """
    from brain_api.storage.local import LSTMHalalNewModelStorage

    local_storage = LSTMHalalNewModelStorage()
    lstm_path = local_storage.lstm_path

    if not lstm_path.exists():
        print("No models found locally")
        return 0

    print(f"Local models in: {lstm_path}")
    print("-" * 60)

    current_version = local_storage.read_current_version()

    # List version directories
    versions = []
    for item in lstm_path.iterdir():
        if item.is_dir() and item.name.startswith("v"):
            versions.append(item.name)

    if not versions:
        print("No model versions found")
        return 0

    versions.sort(reverse=True)

    for version in versions:
        is_current = version == current_version
        marker = " <- current" if is_current else ""

        # Try to load metadata for more info
        metadata = local_storage.read_metadata(version)
        if metadata:
            metrics = metadata.get("metrics", {})
            val_loss = metrics.get("val_loss", "N/A")
            window = metadata.get("data_window", {})
            start = window.get("start", "N/A")
            end = window.get("end", "N/A")
            print(f"  {version}{marker}")
            print(f"    Window: {start} to {end}")
            print(f"    Val Loss: {val_loss}")
        else:
            print(f"  {version}{marker}")

    return 0


def main() -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Push model/dataset artifacts to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Model push command
    model_parser = subparsers.add_parser("model", help="Push LSTM model to HuggingFace")
    model_parser.add_argument(
        "--version",
        required=True,
        help="Model version to push (e.g., 'v2025-12-30-96849e1ed625')",
    )
    model_parser.add_argument(
        "--make-current",
        action="store_true",
        help="Also update main branch to this version",
    )

    # List models command
    subparsers.add_parser("list-models", help="List available local model versions")

    args = parser.parse_args()

    if args.command == "model":
        return push_model(args.version, args.make_current)
    elif args.command == "list-models":
        return list_local_models()
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
