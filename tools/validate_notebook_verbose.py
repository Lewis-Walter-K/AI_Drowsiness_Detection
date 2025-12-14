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
        src_lines = c.get('source', [])
        src = '\n'.join(src_lines)
        try:
            compile(src, f'<cell {i}>', 'exec')
        except Exception as e:
            print('CELL', i, 'ERROR:', e)
            print('--- CELL SOURCE START ---')
            for n, line in enumerate(src_lines, 1):
                print(f'{n:03d}: {line}')
            print('--- CELL SOURCE END ---')
            sys.exit(1)

print('NO SYNTAX ERRORS')
