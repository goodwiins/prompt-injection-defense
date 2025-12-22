import json

nb_path = '/Users/goodwiinz/development/prompt-injection-defense/bit_demonstration from Colab.ipynb'

with open(nb_path, 'r') as f:
    nb = json.load(f)

print("Checking first 10 cells for 'Part 1: Data Composition'...")
for i in range(10):
    cell = nb['cells'][i]
    source = "".join(cell['source'])
    print(f"[{i}] {cell['cell_type']}: {source[:100]}...")
