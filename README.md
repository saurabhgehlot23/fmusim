import csv
import pandas as pd

file_path = "your_file.csv"

with open(file_path, 'r') as f:
    reader = list(csv.reader(f))

# Step 1: Headers (row 3 → index 2, from column C onward)
header_row = reader[2]
headers = ["time"] + header_row[2:]

# Step 2 & 3: Extract data (from row 9 → index 8)
rows = []

for row in reader[8:]:
    if not row:
        continue

    # Time from column B (index 1)
    time_val = float(row[1]) if row[1] else 0.0

    # Data from column C onward
    values = [float(x) if x else 0.0 for x in row[2:]]

    rows.append([time_val] + values)

# Create DataFrame
df = pd.DataFrame(rows, columns=headers)

# Preview
print(df.head())
