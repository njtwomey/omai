import { type ReactNode, useMemo, useState } from 'react'
import { confusionMargins, rocCurve, thresholdConfusion } from 'whitebox-engine/metrics'
import { key, normal } from 'whitebox-engine/random'
import { cartesian, histogram, lines, points, referenceLines } from 'whitebox-render/charts'
import { Chart, ContingencyTable, ControlPanel, FigureFrame, NumberField, useColourMode } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Threshold and contingency',
  question: 'How does the threshold on a classifier’s scores move its contingency table?',
  explain:
    'Scores of two classes overlap; a threshold turns them into decisions. `thresholdConfusion(yTrue, scores, threshold)` gives the 2 by 2 table (positive predicted when the score is at least the threshold), which `ContingencyTable` draws with every rate. Move the threshold to trade false positives for false negatives: the cut moves on the histograms, the operating point moves along the ROC curve, and the table and its rates update together. Separation and prevalence change the problem itself.',
}

const n = 400
const noise = Array.from(normal(key(7), [n]).data)
const order = Array.from(normal(key(8), [n]).data)

export default function ThresholdRecipe(): ReactNode {
  const mode = useColourMode()
  const [threshold, setThreshold] = useState(0.8)
  const [separation, setSeparation] = useState(1.6)
  const [prevalence, setPrevalence] = useState(0.3)
  // Rank a fixed noise sample so prevalence picks the positives deterministically.
  const { y, scores } = useMemo(() => {
    const cut = [...order].sort((a, b) => a - b)[Math.floor((1 - prevalence) * n)] ?? 0
    const y = order.map((v) => (v >= cut ? 1 : 0))
    return { y, scores: noise.map((e, i) => (y[i] === 1 ? separation : 0) + e) }
  }, [separation, prevalence])
  const table = useMemo(() => thresholdConfusion(y, scores, threshold), [y, scores, threshold])
  const roc = useMemo(() => rocCurve(y, scores), [y, scores])
  const m = confusionMargins(table)
  const [fpr, tpr] = [m.rowMiss[0] ?? 0, m.rowRate[1] ?? 0]

  const scoresOption = useMemo(() => {
    const range: [number, number] = [-3.5, separation + 3.5]
    return cartesian(
      [
        histogram(
          scores.filter((_, i) => y[i] === 0),
          { bins: 30, range, name: 'negative', opacity: 0.6, mode },
        ),
        histogram(
          scores.filter((_, i) => y[i] === 1),
          { bins: 30, range, name: 'positive', opacity: 0.6, colourOffset: 1, mode },
        ),
        referenceLines({ x: [{ value: threshold, label: 'threshold' }] }, { colour: 'fg', dash: 'dashed', mode }),
      ],
      { mode, x: { label: 'score' }, y: { label: 'count' } },
    )
  }, [scores, y, threshold, separation, mode])
  const rocOption = useMemo(
    () =>
      cartesian(
        [
          lines([{ x: roc.fpr, y: roc.tpr, name: 'ROC' }], { mode }),
          lines([{ x: [0, 1], y: [0, 1], colour: 'muted', dash: 'dotted', name: 'chance' }], { mode }),
          points([{ x: [fpr], y: [tpr], name: 'operating point', sizes: [1], sizeRange: [14, 14] }], {
            mode,
            colourOffset: 3,
          }),
        ],
        {
          mode,
          x: { label: 'false-positive rate', limits: [0, 1] },
          y: { label: 'true-positive rate', limits: [0, 1] },
        },
      ),
    [roc, fpr, tpr, mode],
  )

  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="threshold" value={threshold} onChange={setThreshold} min={-3} max={5} step={0.05} />
        <NumberField label="separation" value={separation} onChange={setSeparation} min={0} max={4} step={0.1} />
        <NumberField label="prevalence" value={prevalence} onChange={setPrevalence} min={0.05} max={0.95} step={0.05} />
      </ControlPanel>
      <div className="grid gap-3 md:grid-cols-2">
        <FigureFrame filename="threshold-scores">
          <Chart option={scoresOption} />
        </FigureFrame>
        <FigureFrame filename="threshold-roc">
          <Chart option={rocOption} />
        </FigureFrame>
      </div>
      <ContingencyTable table={table} classNames={['negative', 'positive']} positive={1} />
    </div>
  )
}
