import xml.etree.ElementTree as ET
from pathlib import Path


classes = {
    "crazing": 0,
    "inclusion": 1,
    "patches": 2,
    "pitted_surface": 3,
    "rolled-in_scale": 4,
    "scratches": 5,
}


def convert_annotation(xml_path: Path, output_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)

    lines = []

    for obj in root.findall("object"):
        class_name = obj.find("name").text
        class_id = classes[class_name]

        bbox = obj.find("bndbox")

        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height

        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        lines.append(
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{box_width:.6f} "
            f"{box_height:.6f}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def convert_split(split):
    annotation_dir = Path(f"data/NEU-DET/{split}/annotations")
    image_dir = Path(f"data/NEU-DET/{split}/images")
    label_dir = Path(f"data/NEU-DET/{split}/labels")

    for xml_path in annotation_dir.glob("*.xml"):

        image_matches = list(image_dir.rglob(f"{xml_path.stem}.*"))

        if not image_matches:
            print(f"Skipping {xml_path.name}: matching image not found")
            continue

        image_path = image_matches[0]

        relative_image_path = image_path.relative_to(image_dir)

        output_path = label_dir / relative_image_path.with_suffix(".txt")

        convert_annotation(xml_path, output_path)


if __name__ == "__main__":
    convert_split("train")
    convert_split("validation")

    print("Annotation conversion completed.")