from app.graph import vision_node


state = {
    "image_path": "data/NEU-DET/train/images/crazing/crazing_240.jpg",
}

result = vision_node(state)

print("\nVision result:")
print(result)