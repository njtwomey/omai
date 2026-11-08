import { key, normal } from 'omai-engine/random'
import { toArray } from 'omai-engine/tensor'
import { cartesian, ridgeline } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import { Chart, ControlPanel, FigureFrame, NumberField, SeedInput, useColourMode } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Ridgeline plot',
  question: 'How do I compare many distributions at once?',
  explain:
    '`ridgeline(groups, { overlap })` draws one kernel density per group, stacked with a little overlap (a joyplot), so a dozen distributions read at a glance where a dozen histograms would not. Toy daily temperatures by month: the distributions shift and widen through the year.',
}

const FIGURE = figure({ width: 720, height: 560 })
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

export default function Ridgeline(): ReactNode {
  const mode = useColourMode()
  const [overlap, setOverlap] = useState(1.8)
  const [days, setDays] = useState(60)
  const [seed, setSeed] = useState(0)
  const option = useMemo(() => {
    const groups = MONTHS.map((name, m) => {
      const mean = 11 + 8 * Math.sin(((m - 3) * Math.PI) / 6)
      const sd = 2.5 + 1.5 * Math.abs(Math.cos((m * Math.PI) / 6))
      return { name, values: (toArray(normal(key(seed * 12 + m), [days])) as number[]).map((z) => mean + sd * z) }
    })
    return cartesian([ridgeline(groups, { overlap, mode })], {
      mode,
      x: { label: 'daily temperature (°C)' },
      y: { label: '', format: (v) => MONTHS[Math.round(v)] ?? '', splitLine: false },
    })
  }, [overlap, days, seed, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="overlap" value={overlap} onChange={setOverlap} min={0.5} max={4} step={0.1} />
        <NumberField label="days per month" value={days} onChange={setDays} min={5} max={1000} integer log />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="ridgeline">
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
    </div>
  )
}
