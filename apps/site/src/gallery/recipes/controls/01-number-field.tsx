import { type ReactNode, useMemo, useState } from 'react'
import { lineChart } from 'whitebox-render/charts'
import { figure } from 'whitebox-render/figure'
import { Chart, ControlPanel, NumberField, useColourMode } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Number field',
  question: 'How do I take a number: typed, slid, stepped or picked from a list?',
  explain:
    '`NumberField` is one tight row: type a value, drag the rail on its bottom edge, step with the arrows (Shift for ten steps), or open the dropdown of 11 values between `min` and `max`. `integer` and `log` change how it steps, slides and ladders. A typed value that breaks a rule (try 0 for the rate) turns the field red and is not applied.',
}

const FIGURE = figure()

export default function NumberFieldRecipe(): ReactNode {
  const mode = useColourMode()
  const [rate, setRate] = useState(0.05)
  const [steps, setSteps] = useState(60)
  const option = useMemo(() => {
    const t = Array.from({ length: steps + 1 }, (_, i) => i)
    return lineChart([{ x: t, y: t.map((k) => (1 - rate) ** k), name: '(1 − rate)ᵗ' }], {
      mode,
      figure: FIGURE,
      x: { label: 'step t', limits: [0, 100] },
      y: { label: 'remaining', limits: [0, 1] },
    })
  }, [rate, steps, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="decay rate" value={rate} onChange={setRate} min={1e-3} max={1} log />
        <NumberField label="steps shown" value={steps} onChange={setSteps} min={1} max={100} integer />
      </ControlPanel>
      <Chart option={option} figure={FIGURE} />
    </div>
  )
}
