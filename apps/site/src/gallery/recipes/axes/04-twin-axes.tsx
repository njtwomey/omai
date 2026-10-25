import { cartesian, lines } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import { Chart, ControlPanel, FigureFrame, NumberField, useColourMode } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Twin axes',
  question: 'How do I draw two quantities with different units against the same x?',
  explain:
    'Mark a layer `secondary` and it is read against the right-hand axis, `y2`, with its own label and limits. Use it sparingly: two scales invite the reader to compare heights that mean nothing; here temperature and rainfall by month, a common climate chart.',
}

const FIGURE = figure({ width: 820, height: 420 })
const MONTHS = Array.from({ length: 12 }, (_, m) => m + 1)

export default function TwinAxes(): ReactNode {
  const mode = useColourMode()
  const [wet, setWet] = useState(1)
  const option = useMemo(() => {
    const temp = MONTHS.map((m) => 11 + 8 * Math.sin(((m - 4) * Math.PI) / 6))
    const rain = MONTHS.map((m) => wet * (60 + 35 * Math.cos(((m - 1) * Math.PI) / 6)))
    const right = {
      ...lines([{ x: MONTHS, y: rain, name: 'rainfall (mm)', width: 2, dash: 'dashed' }], { mode, colourOffset: 1 }),
      secondary: true,
    }
    return cartesian([lines([{ x: MONTHS, y: temp, name: 'temperature (°C)', width: 2 }], { mode }), right], {
      mode,
      x: { label: 'month', limits: [1, 12] },
      y: { label: 'temperature (°C)' },
      y2: { label: 'rainfall (mm)', limits: [0, 250] },
    })
  }, [wet, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="wetter ×" value={wet} onChange={setWet} min={0.25} max={2.5} step={0.25} />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="twin-axes">
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
    </div>
  )
}
