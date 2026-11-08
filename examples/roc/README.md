# The metrics of a classifier

A classifier gives each case a score, and a decision threshold turns scores into yes or no. The metrics here describe how
good the scores are over every possible threshold (the three curves) and how good one chosen threshold is (the contingency
table).

## What the program does

- It fits a support vector classifier on 120 points of the two-moons data, as a session, so the run has a tape.
- It scores 300 held-out points with the classifier's decision function.
- `mdml-engine/metrics` makes the ROC curve, the precision-recall curve, the precision-recall gain curve and the
  confusion matrix at threshold 0. The render package draws the three curves with its own charts and the table with its own
  `ContingencyTable` component.

## What to look for

- **ROC curve:** the true positive rate against the false positive rate as the threshold moves. The dashed diagonal is
  guessing. The area under the curve is 0.973.
- **Precision-recall curve:** precision against recall. The dashed line is the prevalence (0.5), which is what guessing
  scores. The average precision is 0.975.
- **Precision-recall gain curve:** the same idea with both axes rescaled against that guessing baseline, so that the
  diagonal is again the worst case and the unit square is comparable between datasets.
- **Contingency table:** at threshold 0 the classifier gets 138 + 134 of 300 right. Rows are the actual class and columns
  the predicted class, with the totals, the share of all cases in each cell, and the rates on the margins: sensitivity 0.893,
  specificity 0.92, precision 0.918, negative predictive value 0.896, accuracy 0.907.

To run it again: `make readme-examples ARGS="roc"`. Source: [main.tsx](main.tsx). Figure: [figure.svg](figure.svg).
