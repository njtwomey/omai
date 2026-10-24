import { type ReactNode, useMemo, useState } from 'react'
import { cartesian, contours, heatmap, type Smoothing } from 'whitebox-render/charts'
import { figure } from 'whitebox-render/figure'
import { Chart, Choice, ControlPanel, FigureFrame, NumberField, useColourMode } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Smooth heatmaps',
  question: 'How do I draw a field sampled on a coarse grid without blocky cells?',
  explain:
    '`heatmap(z, { x, y })` upsamples the grid (bicubic by default) into one image, so a 60 by 60 grid costs one image rather than 3600 cells. Coarsen the grid to see how much smoothing recovers. `smooth` picks `bicubic`, `bilinear`, filled `contours` or blocky `cells`; `centre` makes the ramp diverging around a value.',
}

const fig = figure({ width: 560, aspect: 1.25 })
function field(n: number): number[][] {
  const xs = Array.from({ length: n }, (_, j) => -3 + (6 * j) / (n - 1))
  return xs.map((y) => xs.map((x) => Math.sin(x) * Math.cos(y) + 0.3 * Math.exp(-(x * x + y * y) / 2)))
}
const modes: readonly Smoothing[] = ['bicubic', 'bilinear', 'contours', 'cells']

export default function SmoothHeatmap(): ReactNode {
  const mode = useColourMode()
  const [smooth, setSmooth] = useState<Smoothing>('bicubic')
  const [n, setN] = useState(12)
  const option = useMemo(() => {
    const z = field(n)
    return cartesian(
      [
        heatmap(z, { x: [-3, 3], y: [-3, 3], smooth, centre: 0, mode, label: 'f(x, y)' }),
        contours(z, { x: [-3, 3], y: [-3, 3], levels: [0], labels: false, mode, opacity: 0.6 }),
      ],
      { mode, figure: fig, aspect: 'equal' },
    )
  }, [mode, smooth, n])
  return (
    <div className="flex flex-col gap-2">
      <ControlPanel columns={2}>
        <Choice label="smoothing" value={smooth} options={modes} onChange={setSmooth} />
        <NumberField label="grid points per side" value={n} onChange={setN} min={4} max={60} integer />
      </ControlPanel>
      <FigureFrame figure={fig} filename="heatmap">
        <Chart option={option} />
      </FigureFrame>
    </div>
  )
}
