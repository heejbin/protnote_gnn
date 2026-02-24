"""Helper functions for data directory syncing."""

import argparse
import logging
import os
import zipfile
from pathlib import Path

import requests
from huggingface_hub import HfApi, login, snapshot_download
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from protnote.utils.configs import get_project_root, register_resolvers

project_root = get_project_root()
register_resolvers()
GlobalHydra.instance().clear()
with initialize_config_dir(version_base=None, config_dir=str(project_root / "configs")):
    cfg = compose(config_name="config")

datapath_remote = cfg.remote

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _fetch_data_published() -> None:
    """Fetch ProtNote's published data & model from Zenodo."""
    # Download original data from Zenodo
    from tqdm import tqdm

    response = requests.get(datapath_remote["ZENODO_ORIGINAL_DATA"], stream=True)
    response.raise_for_status()
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024

    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

    with open(project_root / "data.zip", "wb") as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)

    progress_bar.close()

    # Unzip the downloaded file to data directory
    with zipfile.ZipFile(project_root / "data.zip", "r") as zip_ref:
        zip_ref.extractall(project_root / "data")
    (project_root / "data.zip").unlink()


def _check_num_files(directory: Path) -> list[str]:
    """Check if any directory has more than 2,000 files at its own level."""
    result = []
    for dirpath, dirnames, filenames in os.walk(directory):
        if len(filenames) > 2000:
            result.append(dirpath)
    return result


def data_upload(repo_id: str, ignore_num_files: bool = False) -> None:
    """Upload data directory to HuggingFace Hub Repository."""
    if not ignore_num_files:
        dirpaths = _check_num_files(project_root / "data")
        if dirpaths:
            raise ValueError(
                f"Directories {', '.join(dirpaths)} have more than 2,000 files. "
                "Compress them before uploading, or you can use --ignore-num-files to bypass."
            )
    api = HfApi()
    api.upload_large_folder(repo_id=repo_id, repo_type="model", folder_path=project_root / "data")


def data_download(repo_id: str, original: bool = False) -> None:
    """Download data and save to the data directory."""
    if original:
        _fetch_data_published()
    else:
        # Download data from HuggingFace Hub Repository
        snapshot_download(repo_id, local_dir=project_root / "data")


if __name__ == "__main__":
    Path(project_root / "data").mkdir(exist_ok=True)

    parser = argparse.ArgumentParser(description="ProtNote data helper")
    parser.add_argument("action", choices=["upload", "download"], help="Action: upload / download")
    parser.add_argument("--original", action="store_true", help="Download original published data")
    parser.add_argument("--repo_id", type=str, default=None, help="HuggingFace repository ID")
    parser.add_argument("--ignore-num-files", action="store_true", help="Ignore number of files check")
    args = parser.parse_args()

    if not args.original:
        # HuggingFace login
        login(new_session=False)

        # Fetch data repo id from config file if not provided
        if args.repo_id is None:
            args.repo_id = datapath_remote["HUGGINGFACE_DATA_REPO"]

    if args.action == "upload":
        data_upload(args.repo_id, args.ignore_num_files)
    elif args.action == "download":
        data_download(args.repo_id, args.original)
