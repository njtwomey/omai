import { waterfallChart } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import { Chart, ControlPanel, FigureFrame, NumberField, useColourMode } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Waterfall chart',
  question: 'How do I show a total built up from rises and falls?',
  explain:
    '`waterfallChart(steps, { start })` floats each step from the running total before it to the total after, rises and falls in different colours, and ends on the total. A toy year of a household budget: change the rent and see the end balance move.',
}

const FIGURE = figure({ width: 820, height: 420 })

export default function Waterfall(): ReactNode {
  const mode = useColourMode()
  const [rent, setRent] = useState(14)
  const option = useMemo(
    () =>
      waterfallChart(
        [
          { name: 'salary', value: 42 },
          { name: 'rent', value: -rent },
          { name: 'food', value: -7 },
          { name: 'travel', value: -4 },
          { name: 'gifts', value: 2 },
          { name: 'bills', value: -5 },
          { name: 'savings interest', value: 1 },
          { name: 'holiday', value: -6 },
        ],
        { mode, start: { name: 'start', value: 5 }, total: 'end', y: { label: 'thousands' } },
      ),
    [rent, mode],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="rent" value={rent} onChange={setRent} min={0} max={40} integer />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="waterfall">
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
    </div>
  )
}
