import { lineChart } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import { Chart, ControlPanel, FigureFrame, NumberField, useColourMode } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Number field',
  question: 'How do I take a number: typed, slid, stepped or picked from a list?',
  explain:
    '`NumberField` is one tight row: type a value, drag the rail on its bottom edge, step with the arrows (Shift for ten steps), or open the dropdown of 11 values between `min` and `max`. `integer` and `log` change how it steps, slides and ladders. A typed value that breaks a rule (try 0 for the rate) turns the field red and is not applied. A number can be shown as words: `format` draws it, `parse` reads what is typed back and `draftText` says what the field holds when it is focused (try typing weak or strong).',
}

const FIGURE = figure()
const LEVELS = ['none', 'weak', 'medium', 'strong'] as const
const levelText = (v: number): string => LEVELS[Math.round(v)] ?? String(v)
const levelOf = (text: string): number | null => {
  const at = LEVELS.indexOf(text.trim().toLowerCase() as (typeof LEVELS)[number])
  return at < 0 ? null : at
}

export default function NumberFieldRecipe(): ReactNode {
  const mode = useColourMode()
  const [rate, setRate] = useState(0.05)
  const [steps, setSteps] = useState(60)
  const [damping, setDamping] = useState(1)
  const option = useMemo(() => {
    const t = Array.from({ length: steps + 1 }, (_, i) => i)
    return lineChart([{ x: t, y: t.map((k) => (1 - rate) ** k * (1 - 0.1 * damping)), name: '(1 − rate)ᵗ' }], {
      mode,
      figure: FIGURE,
      x: { label: 'step t', limits: [0, 100] },
      y: { label: 'remaining', limits: [0, 1] },
    })
  }, [rate, steps, damping, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="decay rate" value={rate} onChange={setRate} min={1e-3} max={1} log />
        <NumberField label="steps shown" value={steps} onChange={setSteps} min={1} max={100} integer />
        <NumberField
          label="damping"
          value={damping}
          onChange={setDamping}
          min={0}
          max={3}
          integer
          ladder={[0, 1, 2, 3]}
          format={levelText}
          parse={levelOf}
          draftText={levelText}
        />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="decay">
        <Chart option={option} />
      </FigureFrame>
    </div>
  )
}
