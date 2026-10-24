import { type ReactNode, useMemo, useState } from 'react'
import { bars, cartesian } from 'whitebox-render/charts'
import { figure } from 'whitebox-render/figure'
import { Chart, Choice, ControlPanel, Toggle, useColourMode } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Bar chart',
  question: 'How do I compare groups of values side by side, stacked, or horizontally?',
  explain:
    '`bars(categories, groups, options)` is a layer of bars: one `{ name, values }` per group, grouped side by side or `stacked`, upright or `horizontal`. Bars stand on a zero baseline, so the value axis always keeps zero in range.',
}

const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
const groups = [
  { name: 'north', values: [12, 15, 9, 14, 18, 21] },
  { name: 'south', values: [8, 11, 13, 10, 9, 12] },
  { name: 'online', values: [5, 7, 10, 16, 20, 26] },
]

// The fixed figure the chart fills (the builders' default size).
const FIGURE = figure()

export default function Bars(): ReactNode {
  const mode = useColourMode()
  const [stacked, setStacked] = useState(false)
  const [orientation, setOrientation] = useState<'upright' | 'horizontal'>('upright')
  const option = useMemo(
    () => cartesian([bars(months, groups, { mode, stacked, horizontal: orientation === 'horizontal' })], { mode }),
    [mode, stacked, orientation],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Choice label="orientation" value={orientation} onChange={setOrientation} options={['upright', 'horizontal']} />
        <Toggle label="stacked" value={stacked} onChange={setStacked} />
      </ControlPanel>
      <Chart option={option} figure={FIGURE} />
    </div>
  )
}
