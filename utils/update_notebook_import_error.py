
import json
import os

file_path = '/Users/goodwiinz/development/prompt-injection-defense/BIT_Mechanism_Colab_Notebook.ipynb'

with open(file_path, 'r') as f:
    nb = json.load(f)

found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "TfidfSVMBaseline" in source:
            # Found the target cell
            
            # Prepare the new code
            new_source = []
            for line in cell['source']:
                new_line = line.replace("TfidfSVMBaseline", "TfidfSvmBaseline")
                new_source.append(new_line)
            
            cell['source'] = new_source
            found = True
            print("Fixed TfidfSVMBaseline import error.")
            break

if not found:
    print("Target cell not found!")
    # Proceed anyway as it might have been fixed manualy or doesn't exist in the expected format? 
    # But for this task I expect it to be there. I'll just exit.
    exit(1)

with open(file_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
