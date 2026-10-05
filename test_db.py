from app.database import get_all_defect_history

history = get_all_defect_history()

for row in history:
    print(row)