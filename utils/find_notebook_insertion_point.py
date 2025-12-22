import json

nb_path = '/Users/goodwiinz/development/prompt-injection-defense/bit_demonstration from Colab.ipynb'

with open(nb_path, 'r') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        sourceLines = cell['source']
        if sourceLines:
            first_line = sourceLines[0].strip()
            print(f"[{i}] CODE: {first_line[:80]}")
            # Check for key variables
            source = "".join(sourceLines)
            if "probs =" in source or "probs=" in source:
                 print(f"    -> FOUND 'probs =' in cell {i}")
            if "val_labels" in source:
                 print(f"    -> FOUND 'val_labels' in cell {i}")
    elif cell['cell_type'] == 'markdown':
        sourceLines = cell['source']
        if sourceLines:
            first_line = sourceLines[0].strip()
            print(f"[{i}] MD:   {first_line[:80]}")
