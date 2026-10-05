from app.graph import graph


result = graph.invoke(
    {
        "image_path": "data/NEU-DET/train/images/crazing/crazing_240.jpg",
        "detections": [],
        "inspection_history": [],
        "inspection_id": "",
        "defect_analysis": {},
        "quality_assessment": {},
        "decision": {},
        "policy_decision": {},
        "pattern_analysis": {},
    }
)


print("\n=== INSPECTION ID ===")
print(result["inspection_id"])

print("\n=== YOLO DETECTIONS ===")
print(result["detections"])

print("\n=== INSPECTION HISTORY ===")
print(result["inspection_history"])

print("\n=== DEFECT ANALYSIS ===")
print(result["defect_analysis"])

print("\n=== QUALITY ASSESSMENT ===")
print(result["quality_assessment"])

print("\n=== PATTERN ANALYSIS ===")
print(result["pattern_analysis"])

print("\n=== DECISION RECOMMENDATION ===")
print(result["decision"])

print("\n=== FINAL POLICY DECISION ===")
print(result["policy_decision"])