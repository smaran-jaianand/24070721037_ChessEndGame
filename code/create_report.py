"""Build the continuous technical report from recorded experiment outputs."""
import json
import csv
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, KeepTogether, Preformatted
)
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_LEFT

R = Path(__file__).resolve().parent.parent
with (R / 'outputs/results.json').open(encoding='utf8') as handle:
    d = json.load(handle)
info = d['data']
m = d['test_metrics']
cm = d['confusion_matrix_rows_actual_cols_predicted_nowin_won']
with (R / 'outputs/model_comparison.csv').open(encoding='utf8') as handle:
    rows = list(csv.DictReader(handle))
top = d['permutation_top10']
f = lambda value: f'{value:.3f}'

font_dir = Path('C:/Windows/Fonts')
for name, filename in [
    ('ReportSerif', 'georgia.ttf'), ('ReportSerifBold', 'georgiab.ttf'),
    ('ReportSans', 'calibri.ttf'), ('ReportSansBold', 'calibrib.ttf'),
    ('ReportMono', 'consola.ttf')
]:
    pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
pdfmetrics.registerFontFamily('ReportSerif', normal='ReportSerif', bold='ReportSerifBold')
pdfmetrics.registerFontFamily('ReportSans', normal='ReportSans', bold='ReportSansBold')
INK = colors.HexColor('#17212B')
MUTED = colors.HexColor('#52616D')
NAVY = colors.HexColor('#173C56')
GRID = colors.HexColor('#D9E0E5')
WIDTH = A4[0] - 104
s = getSampleStyleSheet()
s.add(ParagraphStyle(name='Eyebrow', fontName='ReportSansBold', fontSize=9,
                     leading=12, textColor=NAVY, spaceAfter=12))
s.add(ParagraphStyle(name='TitleX', fontName='ReportSerifBold', fontSize=27,
                     leading=32, textColor=INK, spaceAfter=10))
s.add(ParagraphStyle(name='Subtitle', fontName='ReportSans', fontSize=12,
                     leading=17, textColor=MUTED, spaceAfter=14))
s.add(ParagraphStyle(name='BodyX', fontName='ReportSerif', fontSize=10.5,
                     leading=15.0, textColor=INK, spaceAfter=7.5,
                     allowWidows=0, allowOrphans=0))
s.add(ParagraphStyle(name='CellX', fontName='ReportSans', fontSize=9.8,
                     leading=12.0, textColor=INK))
s.add(ParagraphStyle(name='HeaderCell', fontName='ReportSansBold', fontSize=9.8,
                     leading=12.0, textColor=colors.white))
s.add(ParagraphStyle(name='CaptionX', fontName='ReportSans', fontSize=9,
                     leading=12.2, textColor=MUTED, spaceBefore=5, spaceAfter=11))
s.add(ParagraphStyle(name='TableCaption', parent=s['CaptionX'],
                     fontName='ReportSansBold', textColor=INK,
                     spaceBefore=5, spaceAfter=5, keepWithNext=True))
s.add(ParagraphStyle(name='CodeLabel', parent=s['TableCaption'],
                     spaceBefore=8, spaceAfter=5))
s.add(ParagraphStyle(name='CodeX', fontName='ReportMono', fontSize=9.4,
                     leading=11.6, textColor=INK, leftIndent=9,
                     rightIndent=9, borderPadding=9,
                     backColor=colors.HexColor('#F2F5F7'),
                     spaceBefore=3, spaceAfter=11))
s['Heading1'].fontName = 'ReportSansBold'
s['Heading1'].fontSize = 15
s['Heading1'].leading = 19
s['Heading1'].textColor = INK
s['Heading1'].spaceBefore = 14
s['Heading1'].spaceAfter = 8
s['Heading1'].keepWithNext = True
s['Heading2'].fontName = 'ReportSansBold'
s['Heading2'].fontSize = 11.7
s['Heading2'].leading = 15
s['Heading2'].textColor = INK
s['Heading2'].spaceBefore = 12
s['Heading2'].spaceAfter = 6
s['Heading2'].keepWithNext = True
story = []
_pending_picture = None
_table_index = 0
_table_titles = [
    'Dataset structure and quality checks',
    'Settings tested for each model',
    'Class weight comparison',

]

def p(text):
    story.append(Paragraph(text, s['BodyX']))

def h(text):
    story.append(Paragraph(text, s['Heading1']))

def sub(text):
    story.append(Paragraph(text, s['Heading2']))

def pic(name, width, height):
    global _pending_picture
    picture = Image(str(R / 'outputs' / name))
    max_width = WIDTH if name != 'category_distribution.png' else 430
    scale = min(max_width / picture.imageWidth, 210 / picture.imageHeight)
    picture.drawWidth = picture.imageWidth * scale
    picture.drawHeight = picture.imageHeight * scale
    picture.hAlign = 'CENTER'
    picture.spaceBefore = 7
    _pending_picture = picture

def cap(text):
    global _pending_picture
    story.append(KeepTogether([_pending_picture, Paragraph(text, s['CaptionX'])]))
    _pending_picture = None

def table(data, widths):
    global _table_index
    title = _table_titles[_table_index]
    _table_index += 1
    story.append(Paragraph(f'Table {_table_index}  {title}', s['TableCaption']))
    total = sum(widths)
    actual_widths = [value / total * WIDTH for value in widths]
    numeric = _table_index in (3, 4, 5, 6)
    cells = []
    for row_no, row in enumerate(data):
        prepared = []
        for column_no, value in enumerate(row):
            base = s['HeaderCell'] if row_no == 0 else s['CellX']
            style = ParagraphStyle('local', parent=base,
                                   alignment=2 if numeric and column_no > 0 else TA_LEFT)
            prepared.append(Paragraph(str(value), style))
        cells.append(prepared)
    t = Table(cells, colWidths=actual_widths, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.white, colors.HexColor('#F3F6F8')]),
        ('GRID', (0, 0), (-1, -1), 0.4, GRID),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 9),
        ('RIGHTPADDING', (0, 0), (-1, -1), 9),
    ]))
    t.spaceAfter = 11
    story.append(t)

def code(label, snippet):
    story.append(KeepTogether([
        Paragraph(label, s['CodeLabel']),
        Preformatted(snippet, s['CodeX'], dedent=0)
    ]))

def screenshot(filename, caption):
    image = Image(str(R / 'report/assets' / filename))
    scale = WIDTH / image.imageWidth
    image.drawWidth = WIDTH
    image.drawHeight = image.imageHeight * scale
    image.hAlign = 'CENTER'
    image.spaceBefore = 7
    story.append(KeepTogether([image, Paragraph(caption, s['CaptionX'])]))

def evaluation_screenshots():
    metric = Image(str(R / 'report/assets/final_test_output.png'))
    matrix = Image(str(R / 'report/assets/confusion_matrix_output.png'))
    left_width, right_width = 194, WIDTH - 210
    metric.drawHeight = metric.imageHeight * left_width / metric.imageWidth
    metric.drawWidth = left_width
    matrix.drawHeight = matrix.imageHeight * right_width / matrix.imageWidth
    matrix.drawWidth = right_width
    pair = Table([[metric, matrix]], colWidths=[210, WIDTH - 210], hAlign='CENTER')
    pair.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))
    caption = ('Figure 6. Program output showing the test scores (left) '
               'and confusion matrix (right). Rows are actual labels and columns are predictions.')
    story.append(KeepTogether([pair, Paragraph(caption, s['CaptionX'])]))

story.append(Paragraph('SUPERVISED MACHINE LEARNING',s['Eyebrow']))
story.append(Paragraph('Chess Endgame<br/>Outcome Classification',s['TitleX']))
story.append(Paragraph('King and rook versus king and pawn',s['Subtitle']))
h('Abstract');p(f"In this project, I used machine learning to predict whether White can win a chess endgame. I used the UCI king-rook versus king-pawn dataset, with won and nowin as the two labels. The data contain {info['rows']} positions and {info['features']} category-based inputs. I used 70% of the data for training and 30% for testing. I compared five models by checking five parts of the training data. The highest average F1 selected {d['selected']}. On the held-out test set, accuracy was {f(m['accuracy'])}, precision {f(m['precision'])}, recall {f(m['recall'])}, F1 {f(m['f1'])}, and ROC AUC {f(m['roc_auc'])}. The test confusion matrix contained {cm[0][1]} false wins and {cm[1][0]} missed wins. These scores apply to this dataset and this split.")
h('1 Introduction');p('My aim was to predict whether White can win in a particular endgame setting: White has a king and rook, Black has a king and pawn on a7, and White is to move. The prediction is a binary position label. The experiment does not choose a move, search a game tree, or play a complete game.')
p('I compared five models using the same data. I checked the data, studied the links between the inputs and the result, and compared the models. I then looked at the errors and the inputs that mattered most to the selected model.')
p('I used won as the positive class. A false positive is a wrong win prediction. A false negative is a winning position that the model misses. These labels describe a position, not the next move.')
h('2 Data and Basic Checks');p('I downloaded the data and its notes from UCI [1]. The data file has 36 input columns and one result column. The UCI web page lists 35 inputs, but the data file and its notes list 36. I followed the order given in the notes. The notes do not fully explain the short input names, so I kept them as supplied.')
table([['Measure','Observed'],['Positions / predictors / target',f"{info['rows']} / {info['features']} / 1"],['Missing feature entries',str(info['missing_entries'])],['Duplicate complete rows',str(info['duplicate_rows'])],['Repeated feature rows / conflicting labels',f"{info['duplicate_features']} / {info['conflicting_features']}"],['Class counts',f"won {info['classes']['won']}; nowin {info['classes']['nowin']}"]],[260,200])
p(f"The data contain {info['classes']['won']/info['rows']:.1%} won positions. I found no repeated input rows. The same input row therefore did not appear in both the training and test sets. Similar positions could still appear in both.")
pic('eda.png',460,180);cap('Figure 1. Class counts and the strongest links between inputs and the result, measured using Cramér’s V.')
p(f"The input most strongly linked to the result is {info['top_associations'][0]['feature']} (Cramér’s V = {info['top_associations'][0]['cramers_v']:.3f}). A strong link does not prove that an input causes the result.")
sub('2.1 Checking the input values');p('All inputs are categories. I counted their values and checked for missing entries and unexpected result labels. A numerical outlier rule would not suit these category codes. The notes do not give all allowed values, so I could not fully check every input value.')
pic('category_distribution.png',430,190);cap('Figure 2. Values of rimmx for each result label. I kept the input name supplied with the data.')
p('I used all rows to describe the data and kept all 36 inputs. I used only the training data to choose the model and its settings. Even without repeated rows, some positions may be closely related. My findings apply to this endgame dataset.')
h('3 Building the Models');sub('3.1 Splitting and preparing the data');p(f"I kept 30% of the data for testing and used random seed 42. I kept a similar class balance in both sets. Training has {info['train_counts']['0']} nowin and {info['train_counts']['1']} won positions; testing has {info['test_counts']['0']} nowin and {info['test_counts']['1']} won positions. I used a preparation step that can fill a missing value with the most common category. One-hot encoding turns each category into zero-or-one inputs. These steps learn only from the training rows in each fold. I also set the encoder to handle new categories without stopping.")
p('For Logistic Regression, Decision Tree and Random Forest, I compared normal training with balanced class weights. These weights adjust for the number of rows in each class. I used KNN and Gradient Boosting without class weights. I did not create extra data using SMOTE.')
sub('3.2 Models and settings');p('I started with Logistic Regression, Decision Tree and KNN. I then added Random Forest and Gradient Boosting. I checked all five models on the same five parts of the training data. This is five-fold cross-validation: each part is used once for checking while the other four are used for training. I first used the default settings, then used GridSearchCV to try the settings below and choose the best F1 for won.')
table([['Model','Training search grid'],['Logistic Regression','C = 0.1, 1, 10; weight = none or balanced'],['Decision Tree','Depth = 5, 10, unlimited; leaf size = 1, 3; weight = none or balanced'],['KNN','Neighbors = 3, 5, 11; uniform or distance weighting'],['Random Forest','100 trees; depth = 12 or unlimited; leaf size = 1, 3; weight = none or balanced'],['Gradient Boosting','Trees = 100, 150; learning rate = 0.05, 0.1; depth = 2, 3']],[125,340])
p('Accuracy is the share of all predictions that are correct. Precision tells me how many predicted wins are correct. Recall tells me how many winning positions the model finds. F1 combines precision and recall. ROC-AUC measures how well the model ranks won positions above nowin positions. I kept the test set aside until I chose the model and its settings.')
sub('3.3 Code for training')
p('I used scikit-learn to build the models. The code below splits the data and prepares the inputs. Every model uses the same training set. The preparation steps learn only from the training rows in each fold.')
code('Code 1  Data split and categorical preprocessing', "Xtr, Xte, ytr, yte = train_test_split(\n    X, y, test_size=0.30, random_state=42, stratify=y\n)\npipe = Pipeline([\n    ('imputer', SimpleImputer(strategy='most_frequent')),\n    ('onehot', OneHotEncoder(\n        handle_unknown='ignore', sparse_output=False\n    )),\n    ('model', est)\n])")
p('For each model, I tried the listed settings and saved the average F1 across the five folds. I chose the model with the highest score. The code below shows these steps. The full code and the executed notebook are included with the report.')
screenshot('baseline_tuned_output.png', 'Figure 3. Program output showing F1 before and after tuning for all five models.')
code('Code 2  Cross validation tuning and final selection', "cv = StratifiedKFold(\n    n_splits=5, shuffle=True, random_state=42\n)\nsearch = GridSearchCV(\n    pipe, grid, cv=cv, scoring='f1', refit=True,\n    n_jobs=1, return_train_score=False\n)\nsearch.fit(Xtr, ytr)\n\n# After all five searches have completed:\nselected = max(rows, key=lambda z: z['tuned_cv_f1'])['model']\nfinal = fitted[selected]")
h('4 Model Comparison')
screenshot('training_cv_output.png', 'Figure 4. Average scores from five training folds and the selected model. The output shows four decimal places.')
p('I used these F1 scores to choose the settings and the final model. They may look better because I chose the best settings on the same folds. Fold SD shows how much the scores varied between folds. I used the separate test set to check the selected model.')
sub('4.1 Choosing the final model');table([['Model','Best unweighted CV F1','Best balanced CV F1']]+[[name,f(w['None']),f(w['balanced'])] for name,w in d['weighting'].items()],[165,145,145])
p(f"My highest tuning score was {f(d['models'][d['selected']]['best_cv_f1'])} for {d['selected']}. Its best settings used balanced class weights, no fixed depth limit and at least one row in each leaf. The balanced tree scored slightly higher than the best tree without weights. This small difference does not show that weights will always help.")
h('5 Test Results')
pic('test_results.png',465,190);cap('Figure 5. Confusion matrix and ROC curve for the selected model on the test set.')

p(f"Of {sum(cm[0])+sum(cm[1])} test positions, the model incorrectly predicted won for {cm[0][1]} nowin cases and missed {cm[1][0]} won cases. The confusion matrix shows the correct and wrong predictions using the model’s default decision rule.")
evaluation_screenshots()
sub('5.1 Code for testing and saving')
p('After choosing the model, I used it to predict the test labels. My code measured the scores, saved them in a JSON file, and saved the trained model with its input preparation steps.')
code('Code 3  Held out predictions and metrics', "pred = final.predict(Xte)\nprob = final.predict_proba(Xte)[:, 1]\ncm = confusion_matrix(yte, pred)\nmetrics = {\n    'accuracy': accuracy_score(yte, pred),\n    'precision': precision_score(yte, pred),\n    'recall': recall_score(yte, pred),\n    'f1': f1_score(yte, pred),\n    'roc_auc': roc_auc_score(yte, prob)\n}")
code('Code 4  Recorded results and fitted pipeline', "(OUT / 'results.json').write_text(\n    json.dumps(results, indent=2), encoding='utf8'\n)\njoblib.dump(final, OUT / 'selected_pipeline.joblib')")
h('6 Understanding the Results');sub('6.1 Feature importance');p(f"To see which inputs mattered, I shuffled each input 15 times and checked how much ROC-AUC fell. This is called permutation importance. The largest average decreases were {top[0]['feature']} ({top[0]['mean_auc_decrease']:.3f}), {top[1]['feature']} ({top[1]['mean_auc_decrease']:.3f}) and {top[2]['feature']} ({top[2]['mean_auc_decrease']:.3f}). A larger fall means the model relied more on that input in this test. This does not prove cause and effect. Related inputs may share importance.")
pic('importance.png',465,205);cap('Figure 7. Input importance from shuffling. The bars show the average fall in ROC-AUC, and the thin lines show how much it varied.')
p('I kept the short input names because the UCI notes do not fully explain them. I saved the full importance ranking. I used it to understand the selected model, not to choose a different model after seeing the test results.')
sub('6.2 Limitations');p('I tested one random split of this dataset. Similar positions may appear in both sets, even though no input rows are identical. The scores may therefore be lower on a different group of positions. I used the given labels and did not check them with a separate chess endgame database. I also used the same folds to try settings and select models, which limits the comparison.')
p('My code takes the prepared input categories and predicts won or nowin. It does not read a chessboard, suggest a move or show a winning sequence. I use nowin as given in the dataset. I do not treat it as a separate loss or draw label.')
h('7 Conclusion and Future Work');p(f"I selected {d['selected']} using the average training-fold F1. It achieved a test F1 of {f(m['f1'])} and ROC-AUC of {f(m['roc_auc'])}. It made {cm[0][1]+cm[1][0]} errors in the held-out set. The given inputs worked well on this split. More tests are needed before using the model on other groups of chess positions.")
p('For future work, I would get full input definitions and the board positions. I could then keep similar positions together, check labels and errors with a chess endgame database, and test a new set of positions. I have not carried out these steps yet.')
h('Repeating the Experiment');p('The full code is in code/experiment.py and the executed notebook. It checks the data, compares the models, saves the search results and tests the chosen model. I used random seed 42 so the steps can be repeated. I saved the software versions and the data file checksum, which helps check that the same file is used.')
p('The measured results are saved in outputs/results.json and outputs/model_comparison.csv. The outputs folder also includes every search result, the input rankings and the figures.')
h('Dataset Reference');p('[1] Shapiro, A. Chess (King-Rook vs. King-Pawn). UCI Machine Learning Repository. DOI: 10.24432/C5DK5C. https://archive.ics.uci.edu/dataset/22/chess+king+rook+vs+king+pawn. Dataset and metadata downloaded from UCI; CC BY 4.0.')

def draw_report_page(pdf_canvas, doc):
    pdf_canvas.saveState()
    header_path = R / 'report/assets/institute_header.png'
    header_width = A4[0] - 80
    header_height = header_width * 163 / 963
    pdf_canvas.drawImage(
        str(header_path), 40, A4[1] - 17 - header_height,
        width=header_width, height=header_height,
        preserveAspectRatio=True, mask='auto'
    )
    pdf_canvas.setStrokeColor(GRID)
    pdf_canvas.setLineWidth(0.4)
    pdf_canvas.line(52, 42, A4[0] - 52, 42)
    pdf_canvas.setFont('ReportSans', 8)
    pdf_canvas.setFillColor(MUTED)
    pdf_canvas.drawString(52, 27, 'Supervised learning technical report')
    pdf_canvas.drawRightString(A4[0] - 52, 27, f'Page {doc.page}')
    pdf_canvas.restoreState()

SimpleDocTemplate(
    str(R / 'report/Chess_Endgame_Report.pdf'), pagesize=A4,
    leftMargin=52, rightMargin=52, topMargin=122, bottomMargin=58,
    title='Chess Endgame Outcome Classification',
    subject='Reproducible supervised classification experiment',
    author='',
).build(story, onFirstPage=draw_report_page, onLaterPages=draw_report_page)
print('Continuous report generated')

