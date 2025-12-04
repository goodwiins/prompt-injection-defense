import json

notebook_path = 'production_demo.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if 'from src.detection.ensemble_classifier import EnsembleClassifier' in line:
                new_source.append('from src.detection.ensemble import InjectionDetector\n')
            else:
                new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Updated imports in production_demo.ipynb")
