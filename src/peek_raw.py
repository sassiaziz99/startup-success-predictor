"""
peek_raw.py
Print the raw header line and first 2 data lines of objects.csv,
with NO parsing/interpretation - just to see the file's true structure.
"""

OBJECTS_CSV = r"C:\Users\LENOVO\Desktop\startup-dashboard-data\objects.csv"

with open(OBJECTS_CSV, encoding="utf-8", errors="ignore") as f:
    for i in range(3):
        line = f.readline()
        print(f"--- LINE {i} ---")
        print(line)
        print()