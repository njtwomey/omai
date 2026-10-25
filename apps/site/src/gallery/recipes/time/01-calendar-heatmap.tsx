import { key, normal } from 'mdml-engine/random'
import { toArray } from 'mdml-engine/tensor'
import { calendarHeatmap } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import { Chart, ControlPanel, FigureFrame, NumberField, SeedInput, useColourMode } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Calendar heatmap',
  question: 'How do I show a year of daily values at a glance?',
  explain:
    '`calendarHeatmap(days)` lays a year out as a grid of weeks across and weekdays down, each day coloured by its value, so weekly rhythms and seasons both show. The toy series is daily steps: fewer at weekends, more in summer.',
}

const FIGURE = figure({ width: 860, height: 230 })

export default function CalendarHeatmap(): ReactNode {
  const mode = useColourMode()
  const [weekend, setWeekend] = useState(0.5)
  const [seed, setSeed] = useState(0)
  const option = useMemo(() => {
    const z = toArray(normal(key(seed), [366])) as number[]
    const days: { date: string; value: number }[] = []
    for (let d = 0; d < 365; d++) {
      const date = new Date(Date.UTC(2026, 0, 1 + d))
      const weekday = date.getUTCDay()
      const summer = Math.cos(((d - 190) * 2 * Math.PI) / 365)
      const base = 8000 + 3000 * summer
      const value = Math.max(0, base * (weekday === 0 || weekday === 6 ? weekend : 1) + 1500 * (z[d] ?? 0))
      days.push({ date: date.toISOString().slice(0, 10), value: Math.round(value) })
    }
    return calendarHeatmap(days, { mode, year: 2026, label: 'steps' })
  }, [weekend, seed, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="weekend activity ×" value={weekend} onChange={setWeekend} min={0} max={2} step={0.05} />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="calendar-heatmap">
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
    </div>
  )
}
