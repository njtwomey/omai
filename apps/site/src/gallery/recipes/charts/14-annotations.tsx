import { key, normal } from 'mdml-engine/random'
import { toArray } from 'mdml-engine/tensor'
import { annotations, cartesian, lines } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import { Chart, ControlPanel, FigureFrame, NumberField, Toggle, useColourMode } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Annotations',
  question: 'How do I point out what matters on a chart?',
  explain:
    "`annotations({ arrows, labels, spans })` is a layer of arrows with their labels, free labels and shaded spans of x, combined with the chart's other layers; here it marks the peak, the dip and a period of interest on a toy series. Move the period and the arrows follow the data.",
}

const FIGURE = figure({ width: 820, height: 420 })

export default function Annotations(): ReactNode {
  const mode = useColourMode()
  const [from, setFrom] = useState(40)
  const [show, setShow] = useState(true)
  const option = useMemo(() => {
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
          },
          { mode },
        ),
      )
    return cartesian(layers, { mode, x: { label: 'day', limits: [0, 119] }, y: { label: 'value', limits: [3, 20] } })
  }, [from, show, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="period starts" value={from} onChange={setFrom} min={0} max={99} integer />
        <Toggle label="annotations" value={show} onChange={setShow} />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="annotations">
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
    </div>
  )
}
