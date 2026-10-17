import csv
import json
import statistics

values = []
versions = []
with open("readings.csv") as stream:
    for row in csv.DictReader(stream):
        # rsplit preserves a negative lower bound, e.g. -2-118.
        lower, upper = map(float, row["reference_range"].rsplit("-", 1))
        values.append((float(row["reading"]) - lower) / (upper - lower))
        versions.append(row["sensor_version"])

split = next((i for i in range(1, len(versions)) if versions[i] != versions[0]), len(values) // 2)
delta = statistics.mean(values[split:]) - statistics.mean(values[:split])
answer = "increase" if delta > 0.05 else "decrease" if delta < -0.05 else "stable"
print(json.dumps({"answer": answer}))
