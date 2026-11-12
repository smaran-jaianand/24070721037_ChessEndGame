# Chess Endgame Classification

This folder contains only the UCI king-rook versus king-pawn classification project.

## Folder layout

- `code/` contains the complete experiment script, executed notebook, notebook builder, report builder, presentation source, and dependencies.
- `report/Chess_Endgame_Report.pdf` is the continuous eight-page technical report, written in simple English, including core implementation excerpts, all four supplied output screenshots, the institute header on every page, and the dataset as its only reference.
- `report/Chess_Endgame_Presentation.pptx` contains ten slides and speaker notes for an approximately 11-minute presentation.
- `data/` contains the original UCI archive, raw records and accompanying metadata.
- `outputs/` contains the measured results, every grid-search candidate, figures, categorical associations, permutation importance and fitted pipeline.

The source is UCI Chess King-Rook vs King-Pawn, Shapiro, DOI 10.24432/C5DK5C, CC BY 4.0. The raw file and its names file specify 36 predictors. The landing page lists 35, a discrepancy documented in the report.

## Run the experiment

Use Python 3.14 and the recorded dependency versions. From this `chess-endgame` folder:

```powershell
python -m pip install -r code/requirements.txt
python code/experiment.py
```

The script resolves data and output paths from its own location, so it also works when launched from another working directory. To rebuild and execute the notebook:

```powershell
python code/build_notebook.py
```

The notebook includes the complete Python implementation in sequential cells, followed by the experiment evidence. It can also be opened and run directly. The builder verifies that its code has the same Python syntax tree as the script and checks that the recorded experiment results remain unchanged.

## Rebuild the report

```powershell
python code/create_report.py
```

The report builder reads the recorded experiment outputs, embeds Georgia, Calibri and Consolas from Windows Fonts, and writes the PDF into `report/`. It uses the supplied image in `report/assets/institute_header.png` as the header on every page. ReportLab is required for PDF generation. The presentation source uses the Codex bundled `@oai/artifact-tool` runtime and requires its presentation skill helpers.

## Experiment records

`outputs/results.json` records the source-data SHA-256, versions, data checks, model selection, test metrics and interpretation. The model-comparison CSV and five grid CSV files preserve the training cross-validation results. `outputs/selected_pipeline.joblib` contains the fitted preprocessing and selected estimator.

Model selection uses only the 70% training partition and five-fold cross-validation. The remaining 30% is the held-out test set. The project predicts the supplied binary labels for this specific endgame. Its feature descriptions do not provide arbitrary board-to-feature conversion or move recommendations.
