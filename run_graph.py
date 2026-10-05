from app.graph import graph


result = graph.invoke(
    {
        "image_path": "data/NEU-DET/train/images/crazing/crazing_240.jpg",
        "detections": [],
        "defect_analysis": {},
        "quality_assessment": {},
        "decision": {},
        "policy_decision": {},
    }
)


print("\n=== YOLO DETECTIONS ===")
print(result["detections"])

print("\n=== DEFECT ANALYSIS ===")
print(result["defect_analysis"])

print("\n=== QUALITY ASSESSMENT ===")
print(result["quality_assessment"])

print("\n=== DECISION RECOMMENDATION ===")
print(result["decision"])

print("\n=== FINAL POLICY DECISION ===")
print(result["policy_decision"])