import json, sys, os

nb_path = os.path.join(os.getcwd(), 'collect_data.ipynb')
if not os.path.exists(nb_path):
    print('Notebook not found:', nb_path)
    sys.exit(2)

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

errors = []
for i, c in enumerate(nb.get('cells', []), 1):
    if c.get('cell_type') == 'code':
        src = '\n'.join(c.get('source', []))
        try:
            compile(src, f'<cell {i}>', 'exec')
        except Exception as e:
            errors.append((i, str(e)))

if errors:
    for i, e in errors:
        print(f'CELL {i} SYNTAX ERROR: {e}')
    sys.exit(1)

print('NO SYNTAX ERRORS')
