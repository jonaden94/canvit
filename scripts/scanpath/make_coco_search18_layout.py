"""Build the RESTRUCTURED COCO-Search18 layout that ``canvit/search/data.py`` reads, as symlinks
into a copy in the ORIGINAL release layout (per-task ``images/<task>/``, JSONs in the root).

    python scripts/scanpath/make_coco_search18_layout.py <original root> <new root>

<new root>/images/<name>.jpg    -> <original root>/images/<task>/<name>.jpg   (any task folder)
<new root>/fixations/<f>.json   -> <original root>/<f>.json                   (every fixation JSON)

The release keeps byte-identical copies of an image under the same name in every task folder
that uses it (checked 2026-11-18 for u13879's copy: 3102 files, 2779 names, no name with two
different sizes), so the name alone identifies the image. Refuses -- before writing anything --
a name whose copies differ in size, and a TP JSON naming an image that is not on disk. Writes
nothing outside <new root>; re-running is a no-op on an existing, matching tree.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(src: Path, dst: Path) -> None:
    images: dict[str, Path] = {}
    for task_dir in sorted(p for p in (src / "images").iterdir() if p.is_dir()):
        for img in sorted(task_dir.glob("*.jpg")):
            prev = images.setdefault(img.name, img)
            if prev.stat().st_size != img.stat().st_size:
                raise SystemExit(f"{img.name}: copies differ in size ({prev} vs {img}); the name is ambiguous")
    jsons = sorted(src.glob("coco_search18_fixations_*.json"))
    if not jsons:
        raise SystemExit(f"no coco_search18_fixations_*.json in {src}")
    for js in jsons:
        missing = {tr["name"] for tr in json.loads(js.read_text())} - images.keys()
        if missing:
            raise SystemExit(f"{js.name}: {len(missing)} images not on disk, e.g. {sorted(missing)[0]}")

    for sub, links in (("images", {n: p for n, p in images.items()}), ("fixations", {j.name: j for j in jsons})):
        (dst / sub).mkdir(parents=True, exist_ok=True)
        for name, target in links.items():
            link = dst / sub / name
            if link.is_symlink() and link.resolve() == target.resolve():
                continue
            link.symlink_to(target.resolve())
    has_ta = any("_TA_" in j.name for j in jsons)
    (dst / "README.md").write_text(
        f"COCO-Search18 in the restructured layout canvit reads, as symlinks into {src.resolve()}\n"
        f"(built by canvit scripts/scanpath/make_coco_search18_layout.py).\n\n"
        f"images/: {len(images)} padded 1680x1050 images. fixations/: {', '.join(j.name for j in jsons)}.\n"
        + ("" if has_ta else "\nNo target-absent (TA) data: use --cfg.condition tp.\n"))
    print(f"{dst}: {len(images)} images, {len(jsons)} fixation files, target-absent: {has_ta}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
