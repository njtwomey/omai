import { type ReactNode, useMemo, useState } from 'react'
import { key, normal } from 'whitebox-engine/random'
import { ControlPanel, NumberField, ThresholdExplorer } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Threshold and contingency',
  question: 'How does the threshold on a classifier’s scores move its contingency table?',
  explain:
    'Scores of two classes overlap; a threshold turns them into decisions. `analyseThresholds(yTrue, scores)` cuts the scores at every distinct threshold in one sorted pass and returns a `thresholdSweep` record (the four counts, the rates, ROC and precision-recall points at each), and `ThresholdExplorer` draws it with one threshold shared by every figure. Drag the line on the histograms (or type it) to trade false positives for false negatives: the operating point moves along the ROC and precision-recall curves and the contingency table and its rates update together. Separation and prevalence change the problem itself.',
}

const n = 400
const noise = Array.from(normal(key(7), [n]).data)
const order = Array.from(normal(key(8), [n]).data)

export default function ThresholdRecipe(): ReactNode {
  const [separation, setSeparation] = useState(1.6)
  const [prevalence, setPrevalence] = useState(0.3)
  // Rank a fixed noise sample so prevalence picks the positives deterministically.
  const { y, scores } = useMemo(() => {
    const cut = [...order].sort((a, b) => a - b)[Math.floor((1 - prevalence) * n)] ?? 0
    const y = order.map((v) => (v >= cut ? 1 : 0))
    return { y, scores: noise.map((e, i) => (y[i] === 1 ? separation : 0) + e) }
  }, [separation, prevalence])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="separation" value={separation} onChange={setSeparation} min={0} max={4} step={0.1} />
        <NumberField label="prevalence" value={prevalence} onChange={setPrevalence} min={0.05} max={0.95} step={0.05} />
      </ControlPanel>
      <ThresholdExplorer yTrue={y} scores={scores} initialThreshold={0.8} range={[-3.5, 7.5]} step={0.05} />
    </div>
  )
}
