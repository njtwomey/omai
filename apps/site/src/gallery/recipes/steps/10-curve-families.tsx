import { curveSet } from 'omai-engine/exposition'
import { ControlPanel, ExpositionView, NumberField } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Curves record',
  question:
    'How do I show several curves at once, on log axes, with bands and marked events, revealed as they are computed?',
  explain:
    'A `curves` record is a set of named series over a shared x, each optionally with a spread or a lower and upper band, on a left or right axis, with log scales and marks at chosen x. `CurvesView` draws them and, with its player, reveals them along x. The series here are gradient descent on a badly scaled quadratic at several learning rates: too small crawls, the best one flies, and past the stability limit the loss grows. Raise the condition number and the best rate shrinks, and the log axis shows the geometric rates as straight lines.',
  shows: ['curves', 'CurvesView'],
}

export default function CurvesRecordRecipe(): ReactNode {
  const [condition, setCondition] = useState(20)
  const [steps, setSteps] = useState(60)
  const record = useMemo(() => {
    const x = Array.from({ length: steps }, (_, i) => i)
    const best = 2 / (1 + condition)
    const run = (rate: number): number[] => {
      let a = 1
      let b = 1
      return x.map(() => {
        const loss = 0.5 * (a * a + condition * b * b)
        a -= rate * a
        b -= rate * condition * b
        return Math.min(loss, 1e6)
      })
    }
    return curveSet({
      x,
      series: [0.1, 0.5, 1, 1.5].map((m) => ({ name: `rate ${(m * best).toPrecision(2)}`, y: run(m * best) })),
      xLabel: 'step',
      yLabel: 'loss',
      yLog: true,
      marks: [{ x: Math.round(steps / 2) }],
    })
  }, [condition, steps])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="condition number" value={condition} onChange={setCondition} min={1} max={500} log />
        <NumberField label="steps" value={steps} onChange={setSteps} min={10} max={200} step={10} integer />
      </ControlPanel>
      <ExpositionView record={record} title="Loss at four learning rates" />
    </div>
  )
}
