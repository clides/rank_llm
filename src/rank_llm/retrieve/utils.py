import hashlib
import logging
import os
import re
from pathlib import Path
from urllib.request import urlretrieve

from huggingface_hub import hf_hub_download
from tqdm import tqdm

logger = logging.getLogger(__name__)


# https://gist.github.com/leimao/37ff6e990b3226c2c9670a2cd1e4a6f5
class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


# For large files, we need to compute MD5 block by block. See:
# https://stackoverflow.com/questions/1131220/get-md5-hash-of-big-files-in-python
def compute_md5(file, block_size=2**20):
    m = hashlib.md5()
    with open(file, "rb") as f:
        while True:
            buf = f.read(block_size)
            if not buf:
                break
            m.update(buf)
    return m.hexdigest()


def download_url(
    url, save_dir, local_filename=None, md5=None, force=False, verbose=True
):
    # If caller does not specify local filename, figure it out from the download URL:
    if not local_filename:
        filename = url.split("/")[-1]
        filename = re.sub(
            "\\?dl=1$", "", filename
        )  # Remove the Dropbox 'force download' parameter
    else:
        # Otherwise, use the specified local_filename:
        filename = local_filename

    destination_path = os.path.join(save_dir, filename)

    if verbose:
        print(f"curr_path{os.getcwd()}")
        print(f"Downloading {url} to {destination_path}...")

    # Check to see if file already exists, if so, simply return (quietly) unless force=True, in which case we remove
    # destination file and download fresh copy.
    if os.path.exists(destination_path):
        if verbose:
            print(f"{destination_path} already exists!")
        if not force:
            if verbose:
                print(f"Skipping download.")
            return destination_path
        if verbose:
            print(f"force=True, removing {destination_path}; fetching fresh copy...")
        os.remove(destination_path)

    with TqdmUpTo(
        unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=filename
    ) as t:
        urlretrieve(url, filename=destination_path, reporthook=t.update_to)

    if md5:
        md5_computed = compute_md5(destination_path)
        assert (
            md5_computed == md5
        ), f"{destination_path} does not match checksum! Expecting {md5} got {md5_computed}."

    return destination_path


def get_cache_home():
    custom_dir = os.environ.get("RANK_LLM_CACHE")
    if custom_dir is not None and custom_dir != "":
        print("custom")
        Path(custom_dir).mkdir(exist_ok=True)
        return custom_dir

    default_dir = "retrieve_results"
    Path(default_dir).mkdir(exist_ok=True)
    return default_dir


def download_cached_hits(
    query_name: str,
    force_download: bool = False,
) -> str:
    """
    Download stored retrieved_results from HuggingFace datasets repo.

    Args:
        query_name: query name (eg. "BM25/retrieve_results_arguana_top100.jsonl")
        force_download: If True, ignores cache and re-downloads

    Returns:
        Local path to the downloaded file
    """
    repo_id = "RankLLMData/RankLLM_Data"
    hf_filename = f"retrieve_results/{query_name}"
    cache_dir = get_cache_home()
    simplified_path = f"{cache_dir}/{query_name}"

    if not force_download and os.path.exists(simplified_path):
        print(f"Loading cached results from {simplified_path}")
        return simplified_path

    file_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=hf_filename,
        local_dir=cache_dir,
        force_download=force_download,
    )
    print(f"Downloaded cached results to {file_path}")

    return file_path
