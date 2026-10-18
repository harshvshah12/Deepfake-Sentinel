"""Count local specimens and refuse to invent a download."""

import hashlib
import json
from pathlib import Path

ROOT = Path("dataset")
CLASSES = ("real", "fake")


def main():
    report = {"roots": {}, "note": "Labels are the folder names. No external dataset was downloaded in this run."}
    for name in CLASSES:
        folder = ROOT / name
        files = sorted([p for p in folder.glob("*") if p.is_file()]) if folder.exists() else []
        sample = []
        for path in files[:5]:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            sample.append({"file": path.name, "sha256": digest, "bytes": path.stat().st_size})
        report["roots"][name] = {"count": len(files), "sample_hashes": sample}
    for extra in ("metadata", "manifests", "splits", "quarantine"):
        (ROOT / extra).mkdir(parents=True, exist_ok=True)
    out = ROOT / "manifests" / "local_inventory.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v["count"] if isinstance(v, dict) and "count" in v else v for k, v in report["roots"].items()}))


if __name__ == "__main__":
    main()
