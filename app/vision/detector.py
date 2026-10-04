from ultralytics import YOLO


model = YOLO("models/forgeSight_yolo.pt")


def detect(image_path: str):

    results = model(image_path)

    return results