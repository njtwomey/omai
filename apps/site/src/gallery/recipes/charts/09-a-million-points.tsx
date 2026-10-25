import { normal } from 'mdml-engine/dists'
import { key } from 'mdml-engine/random'
import { toArray } from 'mdml-engine/tensor'
import { cartesian, lines, lttb } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import { Chart, ControlPanel, FigureFrame, NumberField, SeedInput, useColourMode } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'A million points',
  question: 'How do I draw a series far longer than the screen is wide?',
  explain:
    "`lttb(points, threshold)` (largest-triangle-three-buckets) keeps the points that preserve a line's shape (peaks, troughs and turns) and drops the rest, so a million-step random walk draws from a couple of thousand points and still looks right. The count and the time it took are in the title.",
}

const FIGURE = figure({ width: 760, height: 380 })

export default function AMillionPoints(): ReactNode {
  const mode = useColourMode()
  const [n, setN] = useState(1_000_000)
  const [keep, setKeep] = useState(2000)
  const [seed, setSeed] = useState(0)
  const walk = useMemo(() => {
    const steps = toArray(normal(0, 1).sample(key(seed), [n])) as number[]
    let s = 0
    return steps.map((v, i): [number, number] => {
      s += v
      return [i, s]
    })
  }, [n, seed])
  const option = useMemo(() => {
    const t0 = performance.now()
    const kept = lttb(walk, keep)
    const ms = performance.now() - t0
    return cartesian(
      [lines([{ x: kept.map((p) => p[0]), y: kept.map((p) => p[1]), name: 'walk', width: 1 }], { mode })],
      {
        mode,
        title: `${walk.length.toLocaleString()} points drawn from ${kept.length.toLocaleString()} (LTTB in ${ms.toFixed(0)} ms)`,
        x: { label: 'step' },
        y: { label: 'position' },
      },
    )
  }, [walk, keep, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="points" value={n} onChange={setN} min={1000} max={2_000_000} integer log />
        <NumberField label="kept" value={keep} onChange={setKeep} min={50} max={10000} integer log />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="a-million-points">
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
    </div>
  )
}
