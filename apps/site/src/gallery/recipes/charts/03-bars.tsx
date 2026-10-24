import { type ReactNode, useMemo } from 'react'
import { barChart, paretoChart } from 'whitebox-render/charts'
import { figure } from 'whitebox-render/figure'
import { Chart, FigureFrame, useColourMode } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Bar charts',
  question: 'How do I compare groups with bars, error bars, stacks and a Pareto line?',
  explain:
    '`barChart(categories, groups, options)` places groups side by side (or `stacked`, or `horizontal`) and draws `errors` as error bars; value axes keep zero. `paretoChart` sorts the categories and adds the cumulative share on a secondary axis.',
}

const fig = figure({ width: 420, aspect: 1.3 })

export default function Bars(): ReactNode {
  const mode = useColourMode()
  const grouped = useMemo(
    () =>
      barChart(
        ['ridge', 'lasso', 'tree', 'forest'],
        [
          { name: 'train', values: [0.81, 0.79, 0.95, 0.93], errors: [0.02, 0.02, 0.01, 0.01] },
          { name: 'test', values: [0.78, 0.77, 0.7, 0.84], errors: [0.04, 0.05, 0.06, 0.03] },
        ],
        { mode, figure: fig, y: { label: 'accuracy' } },
      ),
    [mode],
  )
  const horizontal = useMemo(
    () =>
      barChart(
        ['small', 'medium', 'large'],
        [
          { name: 'model A', values: [3, 5, 4], errors: [0.4, 0.6, 0.3] },
          { name: 'model B', values: [2, 6, 5], errors: [0.3, 0.5, 0.7] },
          { name: 'model C', values: [4, 4, 6], errors: [0.5, 0.4, 0.4] },
        ],
        { mode, figure: fig, horizontal: true, x: { label: 'score' } },
      ),
    [mode],
  )
  const pareto = useMemo(
    () => paretoChart(['typos', 'timeouts', 'nulls', 'auth', 'other'], [12, 41, 23, 6, 4], { mode, figure: fig }),
    [mode],
  )
  return (
    <div className="flex flex-wrap gap-4">
      <FigureFrame figure={fig} filename="bars">
        <Chart option={grouped} />
      </FigureFrame>
      <FigureFrame figure={fig} filename="bars-horizontal">
        <Chart option={horizontal} />
      </FigureFrame>
      <FigureFrame figure={fig} filename="pareto">
        <Chart option={pareto} />
      </FigureFrame>
    </div>
  )
}
