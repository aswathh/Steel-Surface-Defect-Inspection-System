import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

root = Path(r"C:\Tasks\Steel Surface Defect Inspection System\data\NEU-DET")
out = root.parent / "yolo"
classes = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

for split, ysplit in [("train", "train"), ("validation", "val")]:
    (out / "images" / ysplit).mkdir(parents=True, exist_ok=True)
    (out / "labels" / ysplit).mkdir(parents=True, exist_ok=True)
    count = 0

    for xml_path in (root / split / "annotations").glob("*.xml"):
        r = ET.parse(xml_path).getroot()
        w = float(r.find("size/width").text)
        h = float(r.find("size/height").text)

        lines = []
        for obj in r.findall("object"):
            cid = classes.index(obj.find("name").text)
            b = obj.find("bndbox")
            x1, y1, x2, y2 = [float(b.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax")]
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        stem = xml_path.stem
        cls_folder = stem.rsplit("_", 1)[0]
        img = root / split / "images" / cls_folder / f"{stem}.jpg"
        if not img.exists():
            print("missing image:", img)
            continue

        shutil.copy(img, out / "images" / ysplit / img.name)
        (out / "labels" / ysplit / f"{stem}.txt").write_text("\n".join(lines))
        count += 1

    print(ysplit, count, "images prepared")

yaml_text = f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
yaml_text += "".join(f"  {i}: {c}\n" for i, c in enumerate(classes))
(out / "data.yaml").write_text(yaml_text)
print("data.yaml written")