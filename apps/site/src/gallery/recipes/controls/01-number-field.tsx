import { type ReactNode, useState } from 'react'
import { ControlPanel, NumberField } from 'whitebox-render/react'
import { Line, Plot, plotFrame, sampleCurve } from '../../demo/Plot.tsx'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Number field',
  question: 'How do I take a number: typed, slid, stepped or picked from a list?',
  explain:
    '`NumberField` is one tight row: type a value, drag the rail on its bottom edge, step with the arrows (Shift for ten steps), or open the dropdown of 11 values between `min` and `max`. `integer` and `log` change how it steps, slides and ladders. A typed value that breaks a rule (try 0 for the rate) turns the field red and is not applied.',
}

const frame = plotFrame({ x: [0, 100], y: [0, 1] })

export default function NumberFieldRecipe(): ReactNode {
  const [rate, setRate] = useState(0.05)
  const [steps, setSteps] = useState(60)
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="decay rate" value={rate} onChange={setRate} min={1e-3} max={1} log />
        <NumberField label="steps shown" value={steps} onChange={setSteps} min={1} max={100} integer />
      </ControlPanel>
      <Plot frame={frame} xLabel="step t" yLabel="(1 − rate)ᵗ">
        <Line frame={frame} points={sampleCurve((t) => (1 - rate) ** t, 0, steps, 200)} />
      </Plot>
    </div>
  )
}
