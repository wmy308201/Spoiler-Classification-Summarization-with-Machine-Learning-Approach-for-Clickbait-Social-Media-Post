import pandas as pd
import json

# Read the JSONL file
predictions = []
with open('spoiler_result.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        predictions.append(data)

# Create DataFrame
df = pd.DataFrame(predictions)

# Save as CSV
df.to_csv('task2_submission.csv', index=False)

print(f"Converted {len(predictions)} predictions to CSV")
print("First few rows:")
print(df.head())
