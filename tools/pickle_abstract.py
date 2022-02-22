import json
import pickle
import sys

def get_abstract(d: dict) -> None:
  if isinstance(d, dict):
    return {k: get_abstract(v) for k, v in d.items()}
  else:
    return f"shape={d.shape}; dtype={d.dtype}"

if __name__ == "__main__":
  input_path = sys.argv[1]
  output_path = sys.argv[2]
  with open(input_path, 'rb') as fp:
    inputs = pickle.load(fp)
  outputs = get_abstract(inputs)
  with open(output_path, 'w') as fp:
    json.dump(outputs, fp, indent=4)
