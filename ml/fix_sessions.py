import csv
import random
import os

def fix_sessions(csv_path: str):
    rows = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            label = row['label']
            row['session_id'] = f"kaggle_{label}_{random.randint(1, 5)}"
            rows.append(row)
            
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    fix_sessions("ml/data/landmarks.csv")
    print("Fixed session IDs for training splits.")
