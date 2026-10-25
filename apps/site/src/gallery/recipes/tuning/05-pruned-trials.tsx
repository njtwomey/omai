import { dict, real } from 'mdml-engine/space'
import {
  functionObjective,
  medianPruner,
  nopPruner,
  randomSampler,
  successiveHalvingPruner,
  tune,
} from 'mdml-engine/tune'
import { intermediateChart } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import { Chart, Choice, ControlPanel, FigureFrame, useColourMode } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Pruned trials',
  question: 'How do I show the curves a pruner stopped early?',
  explain:
    '`intermediateChart` draws every value a trial reported along the way: complete trials in colour, pruned ones grey, ending where the pruner stopped them. Each made-up trial reports a curve falling towards its final loss; the median pruner stops a trial worse than the median of earlier trials at the same step, and successive halving keeps only the best fraction at each rung.',
}

const FIG = figure({ width: 640, aspect: 2 })
const space = dict({ lr: real(1e-4, 1, { log: true }), dropout: real(0, 0.7) })
// No training: each trial reports a curve that decays from 2 towards its final loss over 30 steps.
const curve = functionObjective((p, trial) => {
  const { lr, dropout } = p as { lr: number; dropout: number }
  const final = 0.3 + 0.25 * (Math.log10(lr) + 2.3) ** 2 + 1.2 * (dropout - 0.25) ** 2
  const speed = 3 + 6 * Math.abs(Math.log10(lr) + 2)
  for (let step = 1; step <= 30; step++) {
    if (trial.report(step, final + (2 - final) * Math.exp(-step / speed))) return final
  }
  return final
})
const PRUNERS = {
  none: () => nopPruner(),
  median: () => medianPruner({ nStartupTrials: 5, nWarmupSteps: 3 }),
  halving: () => successiveHalvingPruner(),
}
type Pruner = keyof typeof PRUNERS

export default function PrunedRecipe(): ReactNode {
  const mode = useColourMode()
  const [pruner, setPruner] = useState<Pruner>('median')
  const study = useMemo(
    () => tune(curve, space, { sampler: randomSampler(), pruner: PRUNERS[pruner](), nTrials: 40, seed: 2 }).study,
    [pruner],
  )
  const pruned = study.trials.filter((t) => t.state === 'pruned').length
  const steps = study.trials.reduce((s, t) => s + t.intermediate.length, 0)
  const option = useMemo(() => intermediateChart(study, { mode, figure: FIG, label: 'loss', log: true }), [study, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={1}>
        <Choice label="pruner" value={pruner} options={Object.keys(PRUNERS) as Pruner[]} onChange={setPruner} />
      </ControlPanel>
      <FigureFrame
        figure={FIG}
        filename="pruned-trials"
        caption={`${pruned} of ${study.trials.length} trials pruned; ${steps} of ${30 * study.trials.length} steps run.`}
      >
        <Chart option={option} />
      </FigureFrame>
    </div>
  )
}
