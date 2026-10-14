
from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID = "NetherlandsForensicInstitute/rppg-deepfake-detection"

# Download CelebDF real videos and all forensic fake videos.
# Keep the dataset's original labels and folder structure.
download_dir = Path("data/nfi_download")
download_dir.mkdir(parents=True, exist_ok=True)

print("Starting dataset download...")

snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    allow_patterns=[
        "real/CelebDF/*.mp4",
        "fake/*/*.mp4",
    ],
    local_dir=str(download_dir),
)

print("\nDownload complete!")

for label in ("real", "fake"):
    folder = download_dir / label
    videos = list(folder.rglob("*.mp4")) if folder.exists() else []
    print(f"{label}: {len(videos)} videos found across subdirectories")

