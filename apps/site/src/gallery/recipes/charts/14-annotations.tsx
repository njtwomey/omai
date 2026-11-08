import { key, normal } from 'omai-engine/random'
import { toArray } from 'omai-engine/tensor'
import { annotations, cartesian, lines, plotRectOf } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import { Chart, ControlPanel, FigureFrame, NumberField, Toggle, useColourMode } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Annotations',
  question: 'How do I point out what matters on a chart?',
  explain:
    "`annotations({ arrows, labels, spans, regions })` is a layer of arrows with their labels, free labels, shaded spans of x and shaded regions (a rectangle from `x` and `y`, or a band across the plot from one of them), combined with the chart's other layers; here it marks the peak, the dip, a period of interest and a band of normal values on a toy series. Move the period and the arrows follow the data. `plotRectOf(layers, options)` says where the plotting box lands in pixels, for placing your own marks with the chart's own geometry.",
}

const FIGURE = figure({ width: 820, height: 420 })

export default function Annotations(): ReactNode {
  const mode = useColourMode()
  const [from, setFrom] = useState(40)
  const [show, setShow] = useState(true)
  const [band, setBand] = useState(true)
  const { option, box } = useMemo(() => {
    const z = toArray(normal(key(3), [120])) as number[]
    const x = Array.from({ length: 120 }, (_, i) => i)
    const y = x.map((t, i) => 10 + 4 * Math.sin(t / 12) + 0.03 * t + 0.6 * (z[i] ?? 0))
    const peak = y.indexOf(Math.max(...y))
    const dip = y.indexOf(Math.min(...y))
    const layers = [lines([{ x, y, name: 'series', width: 1.75 }], { mode })]
    if (show)
      layers.push(
        annotations(
          {
            arrows: [
              { from: [peak + 12, (y[peak] ?? 0) + 2], to: [peak, y[peak] ?? 0], label: 'peak' },
              { from: [dip + 12, (y[dip] ?? 0) - 2], to: [dip, y[dip] ?? 0], label: 'dip' },
            ],
            spans: [{ x: [from, from + 20], label: 'period of interest' }],
            ...(band ? { regions: [{ y: [9, 13], label: 'normal range', colour: 'tab2' }] } : {}),
          },
          { mode },
        ),
      )
    const options = {
      mode,
      x: { label: 'day', limits: [0, 119] as const },
      y: { label: 'value', limits: [3, 20] as const },
    }
    return { option: cartesian(layers, options), box: plotRectOf(layers, options) }
  }, [from, show, band, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="period starts" value={from} onChange={setFrom} min={0} max={99} integer />
        <Toggle label="annotations" value={show} onChange={setShow} />
        <Toggle label="normal range" value={band} onChange={setBand} />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="annotations">
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
      <p className="text-xs text-muted-foreground">
        plot box (px): x {box.rect.x}, y {box.rect.y}, {box.rect.width} by {box.rect.height}; day {box.ranges.x?.[0]} to{' '}
        {box.ranges.x?.[1]}
      </p>
    </div>
  )
}
