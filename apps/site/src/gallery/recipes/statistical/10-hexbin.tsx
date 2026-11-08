import { key, normal } from 'omai-engine/random'
import { toArray } from 'omai-engine/tensor'
import { cartesian, hexbin, points } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import {
  Chart,
  Choice,
  ControlPanel,
  FigureFrame,
  NumberField,
  SeedInput,
  Toggle,
  useColourMode,
} from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Hexbin',
  question: 'How do I draw a scatter plot of far too many points?',
  explain:
    '`hexbin(x, y, { radius, log })` bins points into hexagons coloured by count, so dense regions read as dense instead of a solid blot. Switch to points to see the overplotting it replaces; log colouring brings out the sparse tails.',
}

const FIGURE = figure({ width: 600, height: 520 })

export default function Hexbin(): ReactNode {
  const mode = useColourMode()
  const [n, setN] = useState(50000)
  const [radius, setRadius] = useState(0.15)
  const [log, setLog] = useState(true)
  const [view, setView] = useState<'hexbin' | 'points'>('hexbin')
  const [seed, setSeed] = useState(0)
  const option = useMemo(() => {
    const z = toArray(normal(key(seed), [2 * n])) as number[]
    // A banana: x standard normal, y bending with x squared.
    const x = Array.from({ length: n }, (_, i) => z[2 * i] ?? 0)
    const y = x.map((v, i) => 0.5 * v * v - 1 + 0.6 * (z[2 * i + 1] ?? 0))
    const layer =
      view === 'hexbin'
        ? hexbin(x, y, { radius, log, mode })
        : points([{ x, y, name: 'points', size: 2, opacity: 0.3 }], { mode })
    return cartesian([layer], {
      mode,
      aspect: 'equal',
      x: { label: 'x', limits: [-4, 4] },
      y: { label: 'y', limits: [-3, 6] },
    })
  }, [n, radius, log, view, seed, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <Choice label="view" value={view} onChange={setView} options={['hexbin', 'points'] as const} />
        <NumberField label="points" value={n} onChange={setN} min={100} max={500000} integer log />
        <NumberField label="hexagon radius" value={radius} onChange={setRadius} min={0.03} max={0.6} step={0.01} log />
        <Toggle label="log counts" value={log} onChange={setLog} />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="hexbin">
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
    </div>
  )
}
