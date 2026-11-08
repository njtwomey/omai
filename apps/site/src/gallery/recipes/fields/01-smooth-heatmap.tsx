import { cartesian, contours, heatmap, type Smoothing } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import { Chart, Choice, ControlPanel, FigureFrame, NumberField, useColourMode } from 'omai-render/react'
import { formatNumber } from 'omai-render/theme'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Smooth heatmaps',
  question: 'How do I draw a field sampled on a coarse grid without blocky cells, and pick out a band of its values?',
  explain:
    '`heatmap(z, { x, y })` upsamples the grid (bicubic by default) into one image, so a 60 by 60 grid costs one image rather than 3600 cells. Coarsen the grid to see how much smoothing recovers. `smooth` picks `bicubic`, `bilinear`, filled `contours` or blocky `cells`; `centre` makes the ramp diverging around a value. Drag the colour bar’s handles to isolate a band of values: `Chart` repaints the image with the values outside the band faded towards the background (the contours stay), so you can see where the field lies within, say, 0.2 of its peak. With filled contours the bar is a set of bands instead: click a band to hide or show it.',
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
  // The band picked on the colour bar (a new chart starts with the whole range).
  const [band, setBand] = useState<readonly [number, number] | undefined>(undefined)
  const z = useMemo(() => field(n), [n])
  const option = useMemo(
    () =>
      cartesian(
        [
          heatmap(z, { x: [-3, 3], y: [-3, 3], smooth, centre: 0, mode, label: 'f(x, y)' }),
          contours(z, { x: [-3, 3], y: [-3, 3], levels: [0], labels: false, mode, opacity: 0.6 }),
        ],
        { mode, figure: fig, aspect: 'equal' },
      ),
    [z, mode, smooth],
  )
  const events = useMemo(
    () => ({
      datarangeselected: (e: unknown) => {
        const s = (e as { selected?: unknown }).selected
        if (Array.isArray(s)) setBand([Math.min(s[0], s[1]), Math.max(s[0], s[1])])
      },
    }),
    [],
  )
  const values = z.flat()
  const inside = band === undefined ? values.length : values.filter((v) => v >= band[0] && v <= band[1]).length
  const caption =
    band === undefined || smooth === 'contours'
      ? smooth === 'contours'
        ? 'Click a band on the colour bar to hide or show it.'
        : 'Drag the colour bar’s handles to isolate a band of values.'
      : `${Math.round((100 * inside) / values.length)}% of the grid lies between ${formatNumber(band[0], { digits: 3 })} and ${formatNumber(band[1], { digits: 3 })}; the rest is faded.`
  return (
    <div className="flex flex-col gap-2">
      <ControlPanel columns={2}>
        <Choice
          label="smoothing"
          value={smooth}
          options={modes}
          onChange={(v) => {
            setSmooth(v)
            setBand(undefined)
          }}
        />
        <NumberField
          label="grid points per side"
          value={n}
          onChange={(v) => {
            setN(v)
            setBand(undefined)
          }}
          min={4}
          max={60}
          integer
        />
      </ControlPanel>
      <FigureFrame figure={fig} filename="heatmap" caption={caption}>
        <Chart option={option} onEvents={events} />
      </FigureFrame>
    </div>
  )
}
