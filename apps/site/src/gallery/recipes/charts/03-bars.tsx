import { type ReactNode, useMemo, useState } from 'react'
import { barChart, paretoChart } from 'whitebox-render/charts'
import { figure } from 'whitebox-render/figure'
import { Chart, ControlPanel, FigureFrame, NumberField, Toggle, useColourMode } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Bar charts',
  question: 'How do I compare groups with bars, error bars, stacks and a Pareto line?',
  explain:
    '`barChart(categories, groups, options)` places groups side by side (or `stacked`, or `horizontal`) and draws `errors` as error bars on each bar; value axes keep zero. `paretoChart` sorts the categories and adds the cumulative share on a secondary axis.',
}

const fig = figure({ width: 440, aspect: 1.25 })
const models = ['ridge', 'lasso', 'tree', 'forest']
const all = [
  { name: 'train', values: [0.81, 0.79, 0.95, 0.93], errors: [0.02, 0.02, 0.01, 0.01] },
  { name: 'test', values: [0.78, 0.77, 0.7, 0.84], errors: [0.04, 0.05, 0.06, 0.03] },
  { name: 'shifted', values: [0.7, 0.69, 0.55, 0.72], errors: [0.05, 0.05, 0.08, 0.04] },
]

export default function Bars(): ReactNode {
  const mode = useColourMode()
  const [groups, setGroups] = useState(2)
  const [stacked, setStacked] = useState(false)
  const [horizontal, setHorizontal] = useState(false)
  const grouped = useMemo(
    () =>
      barChart(models, all.slice(0, groups), {
        mode,
        figure: fig,
        stacked,
        horizontal,
        ...(horizontal ? { x: { label: 'accuracy' } } : { y: { label: 'accuracy' } }),
      }),
    [mode, groups, stacked, horizontal],
  )
  const pareto = useMemo(
    () => paretoChart(['typos', 'timeouts', 'nulls', 'auth', 'other'], [12, 41, 23, 6, 4], { mode, figure: fig }),
    [mode],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="groups" value={groups} onChange={setGroups} min={1} max={3} integer />
        <Toggle label="stacked" value={stacked} onChange={setStacked} />
        <Toggle label="horizontal" value={horizontal} onChange={setHorizontal} />
      </ControlPanel>
      <div className="flex flex-wrap gap-4">
        <FigureFrame figure={fig} filename="bars">
          <Chart option={grouped} />
        </FigureFrame>
        <FigureFrame figure={fig} filename="pareto">
          <Chart option={pareto} />
        </FigureFrame>
      </div>
    </div>
  )
}
