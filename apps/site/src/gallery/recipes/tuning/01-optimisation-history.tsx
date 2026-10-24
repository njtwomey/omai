import { type ReactNode, useMemo, useState } from 'react'
import { choice, dict, real } from 'whitebox-engine/space'
import { functionObjective, randomSampler, tpeSampler, tune } from 'whitebox-engine/tune'
import { historyChart } from 'whitebox-render/charts'
import { figure } from 'whitebox-render/figure'
import { Chart, Choice, ControlPanel, FigureFrame, NumberField, Toggle, useColourMode } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Optimisation history',
  question: 'How do I show how a hyperparameter search progressed?',
  explain:
    '`historyChart` plots each complete trial’s value against its number, and the best value so far as a step line. The study here tunes a made-up validation loss (no model is trained), so switching between random search and TPE shows what a model-based sampler gains.',
}

const FIG = figure({ width: 640, aspect: 2 })
// A made-up validation loss over three hyperparameters, smallest near lr = 0.005 and dropout = 0.25.
const space = dict({ lr: real(1e-4, 1, { log: true }), dropout: real(0, 0.7), act: choice(['relu', 'tanh', 'gelu']) })
const loss = functionObjective((p) => {
  const { lr, dropout, act } = p as { lr: number; dropout: number; act: string }
  return 0.3 + 0.25 * (Math.log10(lr) + 2.3) ** 2 + 1.2 * (dropout - 0.25) ** 2 + (act === 'tanh' ? 0.08 : 0)
})

export default function HistoryRecipe(): ReactNode {
  const mode = useColourMode()
  const [sampler, setSampler] = useState<'random' | 'TPE'>('TPE')
  const [trials, setTrials] = useState(60)
  const [log, setLog] = useState(true)
  const study = useMemo(
    () =>
      tune(loss, space, { sampler: sampler === 'TPE' ? tpeSampler() : randomSampler(), nTrials: trials, seed: 7 })
        .study,
    [sampler, trials],
  )
  const option = useMemo(
    () => historyChart(study, { mode, figure: FIG, label: 'validation loss', log }),
    [study, mode, log],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <Choice label="sampler" value={sampler} options={['random', 'TPE'] as const} onChange={setSampler} />
        <NumberField label="trials" value={trials} onChange={setTrials} min={10} max={120} step={10} />
        <Toggle label="log loss axis" value={log} onChange={setLog} />
      </ControlPanel>
      <FigureFrame figure={FIG} filename="optimisation-history">
        <Chart option={option} />
      </FigureFrame>
    </div>
  )
}
