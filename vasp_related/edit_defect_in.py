import yaml
from pathlib import Path

file_name = "defect_in.yaml"

with open(file_name, "r") as f:
  data = yaml.safe_load(f)

for key in data.keys():
  data[key] = [-2, -1, 0, 1, 2] # Charge states

Path(file_name).write_text(yaml.dump(data, default_flow_style=None))