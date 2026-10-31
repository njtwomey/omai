import { kernelShap, samplingShapley } from 'mdml-engine/explain'
import { featureAttributions } from 'mdml-engine/exposition'
import { tensor } from 'mdml-engine/tensor'
import { Choice, ControlPanel, ExpositionView, ForcePlotView, NumberField, SeedInput } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { CREDIT_FEATURES, creditPredict, creditRows, creditScore } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Attributions of one prediction',
  question: 'How do I show why a model made one prediction: which features pushed it up and which pulled it down?',
  explain:
    'A `featureAttributions` record with one row draws as a waterfall from the expected output to the prediction, one step per feature (`WaterfallView`), or as a force plot with red pushing up from the left and blue down from the right (`ForcePlotView`). With `convergence` (a sampled estimate at growing sample counts) the waterfall plays the estimate settling. The toy credit model has an income-and-debt interaction; Kernel SHAP splits it exactly, sampling Shapley only approaches it.',
  shows: ['featureAttributions', 'ExpositionView', 'WaterfallView', 'ForcePlotView', 'AttributionView'],
}

type Estimator = 'kernel' | 'sampling'
type Style = 'waterfall' | 'force'

export default function AttributionsOfOneRow(): ReactNode {
  const [row, setRow] = useState(3)
  const [seed, setSeed] = useState(1)
  const [estimator, setEstimator] = useState<Estimator>('kernel')
  const [style, setStyle] = useState<Style>('waterfall')
  const record = useMemo(() => {
    const rows = creditRows(seed, 61)
    const background = rows.slice(20)
    const x = rows[row] ?? []
    const base = background.reduce((a, r) => a + creditScore(r), 0) / background.length
    const common = {
      features: CREDIT_FEATURES,
      output: 'log odds of repaying',
      baseValue: base,
      data: [x],
      predictions: [creditScore(x)],
    }
    if (estimator === 'kernel') {
      const shap = kernelShap(creditPredict, tensor([x]), tensor(background))
      return featureAttributions({ method: 'kernel', ...common, baseValue: shap.baseValue, values: shap.values })
    }
    const counts = [2, 5, 10, 20, 50, 100, 200]
    const runs = counts.map((n) => samplingShapley(creditPredict, x, tensor(background), { nPermutations: n }))
    const last = runs[runs.length - 1]
    return featureAttributions({
      method: 'sampling',
      ...common,
      values: [last?.values ?? []],
      standardErrors: [last?.standardErrors ?? []],
      convergence: {
        row: 0,
        samples: counts,
        estimates: runs.map((r) => r.values),
        standardErrors: runs.map((r) => r.standardErrors),
      },
    })
  }, [seed, estimator, row])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={4}>
        <SeedInput value={seed} onChange={setSeed} />
        <NumberField label="applicant" value={row} onChange={setRow} min={0} max={19} integer />
        <Choice label="estimator" value={estimator} options={['kernel', 'sampling'] as const} onChange={setEstimator} />
        <Choice label="view" value={style} options={['waterfall', 'force'] as const} onChange={setStyle} />
      </ControlPanel>
      {style === 'waterfall' ? (
        <ExpositionView record={record} title="Why this applicant scored as they did" />
      ) : (
        <ForcePlotView record={record} title="Pushes up and pulls down" />
      )}
    </div>
  )
}
