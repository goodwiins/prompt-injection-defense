import json

nb_path = '/Users/goodwiinz/development/prompt-injection-defense/bit_demonstration from Colab.ipynb'

with open(nb_path, 'r') as f:
    nb = json.load(f)

indices_to_check = [11, 57, 65]
for i in indices_to_check:
    print(f"--- Cell {i} ---")
    source = "".join(nb['cells'][i]['source'])
    print(source)
    print("----------------")
