import { barChart, funnelChart, gaugeChart, pieChart } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import { Chart, Choice, ControlPanel, FigureFrame, NumberField, useColourMode } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Pies, funnels and gauges',
  question: 'How do I draw a pie, a donut, a funnel or a gauge, and when should I use bars instead?',
  explain:
    '`pieChart(parts, { inner })` draws a pie or, with an inner radius, a donut; `funnelChart(stages)` the shares kept from stage to stage; `gaugeChart(value, { range })` one reading on a dial. Switch the view to bars: the same five parts, nearly equal, are easy to rank as bars and hard to rank as slices, because we judge lengths better than angles.',
}

const FIGURE = figure({ width: 560, height: 440 })
type View = 'pie' | 'donut' | 'bars' | 'funnel' | 'gauge'
const PARTS = ['north', 'south', 'east', 'west', 'centre']

export default function PiesFunnelsGauges(): ReactNode {
  const mode = useColourMode()
  const [view, setView] = useState<View>('donut')
  const [spread, setSpread] = useState(0.1)
  const option = useMemo(() => {
    const parts = PARTS.map((name, i) => ({
      name,
      value: Math.round(100 * (1 + spread * (i - 2) * (i % 2 === 0 ? 1 : -0.6))),
    }))
    if (view === 'pie' || view === 'donut') return pieChart(parts, { mode, inner: view === 'donut' ? 0.55 : 0 })
    if (view === 'bars')
      return barChart(
        PARTS,
        parts.map((p) => p.value),
        { mode, y: { label: 'sales' } },
      )
    if (view === 'funnel')
      return funnelChart(
        [
          { name: 'visited', value: 10000 },
          { name: 'browsed', value: Math.round(6000 * (1 - spread)) },
          { name: 'basket', value: Math.round(1500 * (1 - spread)) },
          { name: 'checkout', value: Math.round(700 * (1 - spread)) },
          { name: 'paid', value: Math.round(520 * (1 - spread)) },
        ],
        { mode },
      )
    return gaugeChart(Math.round(100 * (0.5 + spread)), { mode, name: 'load', unit: '%' })
  }, [view, spread, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Choice
          label="view"
          value={view}
          onChange={setView}
          options={['pie', 'donut', 'bars', 'funnel', 'gauge'] as View[]}
        />
        <NumberField label="spread" value={spread} onChange={setSpread} min={0} max={0.45} step={0.01} />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="pies-funnels-and-gauges">
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
    </div>
  )
}
