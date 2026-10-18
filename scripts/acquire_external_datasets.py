"""Official dataset access is gated. This script records the blocker instead of guessing files."""

from pathlib import Path

SOURCES = [
    ("FaceForensics++", "https://github.com/ondyari/FaceForensics", "Requires the authors' access form and a download script."),
    ("DFDC", "https://ai.meta.com/datasets/dfdc/", "Requires Meta's dataset terms and a local download."),
    ("Celeb-DF", "https://github.com/yuezunli/celeb-deepfakeforensics", "Requires the authors' release agreement."),
    ("SynthForensics", "https://huggingface.co/datasets/SynthForensics/SynthForensics", "Hugging Face dataset. Not downloaded in this run."),
    ("DeepfakeBench", "https://github.com/SCLBD/DeepfakeBench", "Benchmark harness. Not installed in this run."),
]


def main():
    lines = ["# External datasets", "", "No external archive was downloaded. Local folder labels are the only labels on disk.", ""]
    for name, url, note in SOURCES:
        lines.append(f"- {name}: {url} — {note}")
    lines.append("")
    lines.append("Do not label an image by appearance. A file enters dataset/real or dataset/fake only with a source record in dataset/metadata.")
    Path("dataset/metadata").mkdir(parents=True, exist_ok=True)
    Path("dataset/metadata/EXTERNAL_DATASETS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote dataset/metadata/EXTERNAL_DATASETS.md")


if __name__ == "__main__":
    main()
