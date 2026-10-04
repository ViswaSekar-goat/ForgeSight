from app.graph import graph


result = graph.invoke(
    {
        "image_path": "data/NEU-DET/train/images/crazing/crazing_240.jpg",
        "detections": [],
        "defect_analysis": {},
    }
)


print("\nFINAL STATE:")
print(result)