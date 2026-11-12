"""Create an executed notebook containing the complete experiment source."""
from pathlib import Path
import ast
import json
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
source_path = ROOT / 'code/experiment.py'
source = source_path.read_text(encoding='utf8')
# Split only at top-level statement boundaries. The resulting code is the
# same implementation as the Python script, rather than a hidden run command.
markers = [
    ('Imports', 'from pathlib import Path'),
    ('Data loading and quality checks', 'ROOT = Path'),
    ('Categorical exploration', '# Full-data exploration'),
    ('Model definitions and parameter grids', 'cv = StratifiedKFold'),
    ('Training cross validation and tuning', 'rows = []'),
    ('Selection and held out evaluation', 'selected = max'),
    ('Recorded results and figures', 'results = {'),
]
starts = [source.index(marker) for _, marker in markers]
chunks = [source[start:(starts[i + 1] if i + 1 < len(starts) else len(source))].strip() for i, start in enumerate(starts)]
assert ast.dump(ast.parse('\n\n'.join(chunks))) == ast.dump(ast.parse(source))
nb = nbformat.v4.new_notebook()
nb.cells = [
    nbformat.v4.new_markdown_cell(
        '# Chess endgame classification\n\n'
        'The complete experiment appears below as readable code cells. '
        'The same implementation is available in `experiment.py`. '
        'Model selection uses only training cross-validation. The final '
        'test set evaluates the selected pipeline.'
    ),
    nbformat.v4.new_code_cell(
        'from pathlib import Path\n'
        'import os\n'
        'PROJECT_ROOT = Path.cwd()\n'
        'if PROJECT_ROOT.name == "code":\n'
        '    PROJECT_ROOT = PROJECT_ROOT.parent\n'
        'assert (PROJECT_ROOT / "data/kr-vs-kp.data").exists()\n'
        'os.chdir(PROJECT_ROOT)\n'
        '__file__ = str(PROJECT_ROOT / "code/experiment.py")'
    ),
]
for (title, _), chunk in zip(markers, chunks):
    nb.cells.extend([
        nbformat.v4.new_markdown_cell('## ' + title),
        nbformat.v4.new_code_cell(chunk),
    ])
nb.cells.extend([
    nbformat.v4.new_markdown_cell('## Experiment evidence'),
    nbformat.v4.new_code_cell(
        'from IPython.display import display, Image\n'
        'r = json.loads((OUT / "results.json").read_text())\n'
        'display(pd.read_csv(OUT / "model_comparison.csv"))\n'
        'print("Selected model:", r["selected"])\n'
        'print("Test metrics:", r["test_metrics"])\n'
        'print("Class weight comparison:", r["weighting"])\n'
        'for figure in ["eda.png", "category_distribution.png", '\
        '"test_results.png", "importance.png"]:\n'
        '    display(Image(filename=str(OUT / figure)))\n'
        'display(pd.read_csv(OUT / "permutation_importance.csv").head(10))'
    ),
])
nb.metadata = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3'},
}
previous = json.loads((ROOT / 'outputs/results.json').read_text())
NotebookClient(nb, timeout=900, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}}).execute()
current = json.loads((ROOT / 'outputs/results.json').read_text())
assert current == previous, 'The reorganized code changed the recorded experiment results'
nbformat.write(nb, ROOT / 'code/Chess_Endgame_Experiment.ipynb')
print('Executed full-code notebook saved. Recorded experiment results unchanged.')
