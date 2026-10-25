import { type ReactNode, useMemo, useState } from 'react'
import { key, normal } from 'whitebox-engine/random'
import { ControlPanel, NumberField, ThresholdExplorer } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Misclassification costs',
  question: 'How do I choose a threshold when the two kinds of error cost different amounts?',
  explain:
    'Give each decision a cost — `costs[actual][predicted]`, here a missed case (false negative) and a false alarm (false positive), with correct decisions free. `analyseThresholds(yTrue, scores, { costs })` totals the cost at every threshold in one sorted pass and marks the cheapest; `ThresholdExplorer` follows the cheapest threshold until you drag it (on the cost curve or the histograms), and the contingency table shows each cell’s cost and the total and per-case cost. Make misses dearer and the cheapest threshold drops, buying sensitivity with false alarms; rare positives (low prevalence) push the other way. For calibrated probabilities the cheapest cut is cost(FP) / (cost(FP) + cost(FN)); for raw scores, sweep.',
}

const n = 600
const noise = Array.from(normal(key(21), [n]).data)
const order = Array.from(normal(key(22), [n]).data)
const separation = 2

export default function CostsRecipe(): ReactNode {
  const [missCost, setMissCost] = useState(20)
  const [alarmCost, setAlarmCost] = useState(1)
  const [prevalence, setPrevalence] = useState(0.1)
  const { y, scores } = useMemo(() => {
    const cut = [...order].sort((a, b) => a - b)[Math.floor((1 - prevalence) * n)] ?? 0
    const y = order.map((v) => (v >= cut ? 1 : 0))
    return { y, scores: noise.map((e, i) => (y[i] === 1 ? separation : 0) + e) }
  }, [prevalence])
  const costs = useMemo(
    () => [
      [0, alarmCost],
      [missCost, 0],
    ],
    [missCost, alarmCost],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="cost of a miss (FN)" value={missCost} onChange={setMissCost} min={0} max={100} step={1} />
        <NumberField
          label="cost of a false alarm (FP)"
          value={alarmCost}
          onChange={setAlarmCost}
          min={0}
          max={100}
          step={1}
        />
        <NumberField label="prevalence" value={prevalence} onChange={setPrevalence} min={0.02} max={0.5} step={0.02} />
      </ControlPanel>
      <ThresholdExplorer
        yTrue={y}
        scores={scores}
        costs={costs}
        followCheapest
        figures={['cost', 'scores']}
        range={[-3, 5.5]}
        step={0.05}
      />
    </div>
  )
}
