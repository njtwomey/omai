import { analyseThresholds } from 'omai-engine/metrics'
import { ControlPanel, ExpositionView, NumberField, SeedInput } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { normals, uniforms } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Threshold sweep',
  question: 'How do I show every operating point of a scorer at once, without choosing a threshold?',
  explain:
    'A `thresholdSweep` record is the confusion counts and rates of a scorer at every distinct threshold, with its ROC and precision-recall points. `ThresholdSweepView` is the explorer over it: the scores of the two classes as histograms with a threshold to drag, the ROC and precision-recall curves with the operating point, and the contingency table with its rates, all moving together. The evaluation section’s page builds the same record from raw scores; this one hands the view the record. Separation moves the two score distributions apart and prevalence changes how many positives there are.',
  shows: ['thresholdSweep', 'ThresholdSweepView'],
}

export default function ThresholdSweepRecipe(): ReactNode {
  const [separation, setSeparation] = useState(2)
  const [prevalence, setPrevalence] = useState(0.3)
  const [seed, setSeed] = useState(1)
  const record = useMemo(() => {
    const n = 1500
    const u = uniforms(seed, n)
    const z = normals(seed + 1, n)
    const y = u.map((v) => (v < prevalence ? 1 : 0))
    return analyseThresholds(
      y,
      z.map((v, i) => v + separation * (y[i] ?? 0)),
    )
  }, [separation, prevalence, seed])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="separation" value={separation} onChange={setSeparation} min={0} max={5} step={0.25} />
        <NumberField label="prevalence" value={prevalence} onChange={setPrevalence} min={0.05} max={0.95} step={0.05} />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <ExpositionView record={record} title="Rates at every threshold" />
    </div>
  )
}
