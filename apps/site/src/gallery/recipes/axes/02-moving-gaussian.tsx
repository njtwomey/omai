import { normal } from 'omai-engine/dists'
import { density, type Panel } from 'omai-render/charts'
import { figure, gridSpec, type LimitPolicy, type Range } from 'omai-render/figure'
import { ControlPanel, FigureFrame, FigureGrid, NumberField, useColourMode } from 'omai-render/react'
import { type ReactNode, useMemo, useRef, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Fixed versus fitted limits',
  question: 'How do I make a curve visibly move when its parameter changes?',
  explain:
    'With `auto` limits the axes refit the data, so moving the mean moves the axis instead of the curve. `fixed` limits hold the range, so the curve visibly moves; `expand` grows to cover where it has been; `follow` keeps a window around the latest data. Each panel has its own x policy.',
}

const fig = figure({ width: 680, aspect: 1.5 })
const grid = gridSpec(2, 2, {
  cells: [
    { id: 'fixed', row: 0, col: 0 },
    { id: 'auto', row: 0, col: 1 },
    { id: 'expand', row: 1, col: 0 },
    { id: 'follow', row: 1, col: 1 },
  ],
})
const policies: Readonly<Record<string, LimitPolicy>> = {
  fixed: { kind: 'fixed', range: [-6, 6] },
  auto: { kind: 'auto' },
  expand: { kind: 'expand' },
  follow: { kind: 'follow', window: 8, lead: 2 },
}

export default function MovingGaussian(): ReactNode {
  const mode = useColourMode()
  const [mu, setMu] = useState(0)
  const [sigma, setSigma] = useState(1)
  const seen = useRef<Range | undefined>(undefined)
  const panels = useMemo<Record<string, Panel>>(() => {
    const curve = density(normal(mu, sigma), { mode, fill: 0.2, range: [mu - 4 * sigma, mu + 4 * sigma] })
    const x = curve.extent.x ?? [0, 1]
    seen.current = seen.current === undefined ? x : [Math.min(seen.current[0], x[0]), Math.max(seen.current[1], x[1])]
    return Object.fromEntries(
      Object.entries(policies).map(([id, limits]) => [
        id,
        {
          layers: [curve],
          options: {
            title: id,
            x: { limits },
            y: { limits: [0, 0.45] },
            ...(id === 'expand' && seen.current !== undefined ? { previous: { x: seen.current } } : {}),
          },
        },
      ]),
    )
  }, [mu, sigma, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="mean μ" value={mu} onChange={setMu} min={-4} max={4} />
        <NumberField label="std σ" value={sigma} onChange={setSigma} min={0.3} max={3} />
      </ControlPanel>
      <FigureFrame figure={fig}>
        <FigureGrid figure={fig} grid={grid} panels={panels} />
      </FigureFrame>
    </div>
  )
}
