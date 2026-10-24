import { type ReactNode, useMemo, useState } from 'react'
import { cartesian, contours, heatmap, type Smoothing } from 'whitebox-render/charts'
import { figure } from 'whitebox-render/figure'
import { Chart, FigureFrame, useColourMode } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Smooth heatmaps',
  question: 'How do I draw a field sampled on a coarse grid without blocky cells?',
  explain:
    '`heatmap(z, { x, y })` upsamples the grid (bicubic by default) into one image, so a 25 by 25 grid costs one image rather than 625 cells. `smooth` picks `bicubic`, `bilinear`, filled `contours` or blocky `cells`; `centre` makes the ramp diverging around a value.',
}

const fig = figure({ width: 560, aspect: 1.25 })
const n = 25
const xs = Array.from({ length: n }, (_, j) => -3 + (6 * j) / (n - 1))
const z = xs.map((y) => xs.map((x) => Math.sin(x) * Math.cos(y) + 0.3 * Math.exp(-(x * x + y * y) / 2)))
const modes: readonly Smoothing[] = ['bicubic', 'bilinear', 'contours', 'cells']

export default function SmoothHeatmap(): ReactNode {
  const mode = useColourMode()
  const [smooth, setSmooth] = useState<Smoothing>('bicubic')
  const option = useMemo(
    () =>
      cartesian(
        [
          heatmap(z, { x: [-3, 3], y: [-3, 3], smooth, centre: 0, mode, label: 'f(x, y)' }),
          contours(z, { x: [-3, 3], y: [-3, 3], levels: [0], labels: false, mode, opacity: 0.6 }),
        ],
        { mode, figure: fig, aspect: 'equal' },
      ),
    [mode, smooth],
  )
  return (
    <div className="flex flex-col gap-2">
      <div className="flex gap-2 text-sm">
        {modes.map((m) => (
          <button
            key={m}
            type="button"
            onClick={() => setSmooth(m)}
            className={`rounded border px-2 py-0.5 ${m === smooth ? 'border-[var(--wb-accent)]' : 'border-[var(--wb-grid)]'}`}
          >
            {m}
          </button>
        ))}
      </div>
      <FigureFrame figure={fig} filename="heatmap">
        <Chart option={option} />
      </FigureFrame>
    </div>
  )
}
