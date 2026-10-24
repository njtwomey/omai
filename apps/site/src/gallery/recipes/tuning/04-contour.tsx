import { type ReactNode, useMemo, useState } from 'react'
import { dict, real } from 'whitebox-engine/space'
import { functionObjective, randomSampler, tune } from 'whitebox-engine/tune'
import { contourChart } from 'whitebox-render/charts'
import { figure } from 'whitebox-render/figure'
import { Chart, ControlPanel, FigureFrame, NumberField, Select, useColourMode } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Contour of a study',
  question: 'How do I show how two hyperparameters interact?',
  explain:
    '`contourChart` fits a Gaussian-process surrogate to the trials over two parameters (`contourData` in whitebox-engine’s `tune`) and draws its prediction as a smooth field with contour lines, the trials on top; log-scaled parameters sit on log axes. Here the best learning rate depends on the momentum, so the valley runs diagonally.',
}

const FIG = figure({ width: 560, aspect: 1.3 })
const space = dict({ lr: real(1e-4, 1, { log: true }), momentum: real(0, 0.99), dropout: real(0, 0.7) })
const loss = functionObjective((p) => {
  const { lr, momentum, dropout } = p as { lr: number; momentum: number; dropout: number }
  // The effective step lr / (1 − momentum) wants to be about 0.03.
  return 0.3 + 0.2 * (Math.log10(lr / (1 - momentum)) + 1.5) ** 2 + 0.6 * (dropout - 0.25) ** 2
})
const PAIRS = ['lr × momentum', 'lr × dropout', 'momentum × dropout']

export default function ContourRecipe(): ReactNode {
  const mode = useColourMode()
  const [pair, setPair] = useState('lr × momentum')
  const [trials, setTrials] = useState(50)
  const study = useMemo(() => tune(loss, space, { sampler: randomSampler(), nTrials: trials, seed: 5 }).study, [trials])
  const option = useMemo(() => {
    const [x = 'lr', y = 'momentum'] = pair.split(' × ')
    return contourChart(study, x, y, { mode, figure: FIG, label: 'loss' })
  }, [study, pair, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Select label="parameters" value={pair} options={PAIRS} onChange={setPair} />
        <NumberField label="trials" value={trials} onChange={setTrials} min={10} max={120} step={10} />
      </ControlPanel>
      <FigureFrame figure={FIG} filename="study-contour">
        <Chart option={option} />
      </FigureFrame>
    </div>
  )
}
