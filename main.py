from app.vision.detector import detect


image_path = "data/NEU-DET/validation/images/patches/patches_254.jpg"

results = detect(image_path)

for result in results:

    print("RESULT OBJECT:")
    print(result)

    print("\nBOXES:")
    print(result.boxes)

    print("\nCLASS IDS:")
    print(result.boxes.cls)

    print("\nCONFIDENCE:")
    print(result.boxes.conf)

    print("\nCOORDINATES:")
    print(result.boxes.xyxy)

    print("\nCLASS NAMES:")
    print(result.names)