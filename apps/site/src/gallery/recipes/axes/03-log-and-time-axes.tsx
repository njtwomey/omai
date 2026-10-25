import { cartesian, lines } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import { Chart, Choice, ControlPanel, FigureFrame, NumberField, useColourMode } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Log and time axes',
  question: 'How do I put an axis on a log scale, or on dates?',
  explain:
    "Give an axis `scale: 'log'` and exponential growth becomes a straight line whose slope is the rate, while a linear axis hides everything but the last few days; `scale: 'time'` takes x values as milliseconds and labels them as dates. Toy case counts doubling every few days.",
}

const FIGURE = figure({ width: 820, height: 420 })

export default function LogAndTimeAxes(): ReactNode {
  const mode = useColourMode()
  const [scale, setScale] = useState<'linear' | 'log'>('log')
  const [doubling, setDoubling] = useState(3)
  const option = useMemo(() => {
    const start = Date.UTC(2026, 2, 1)
    const day = 86400000
    const x = Array.from({ length: 60 }, (_, d) => start + d * day)
    const y = x.map((_, d) => 10 * 2 ** (d / doubling))
    const slower = x.map((_, d) => 10 * 2 ** (d / (2 * doubling)))
    return cartesian(
      [
        lines(
          [
            { x, y, name: `doubling every ${doubling} days`, width: 2 },
            { x, y: slower, name: `every ${2 * doubling} days`, width: 2 },
          ],
          { mode },
        ),
      ],
      { mode, x: { label: 'date', scale: 'time' }, y: { label: 'cases', scale } },
    )
  }, [scale, doubling, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Choice label="y scale" value={scale} onChange={setScale} options={['linear', 'log'] as const} />
        <NumberField label="doubling time (days)" value={doubling} onChange={setDoubling} min={1} max={14} step={0.5} />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="log-and-time-axes">
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
    </div>
  )
}
