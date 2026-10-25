import { lines, subplots } from 'mdml-render/charts'
import { figure, gridSpec, type LimitPolicy, type Range } from 'mdml-render/figure'
import { Chart, ControlPanel, FigureFrame, NumberField, Toggle, useColourMode } from 'mdml-render/react'
import { type ReactNode, useEffect, useMemo, useRef, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Limit policies',
  question: 'How do I choose how an axis range follows changing data?',
  explain:
    'A reactive figure picks a `LimitPolicy` per axis: `auto` fits the data, `fixed` holds a stated range (so movement is visible), `expand` grows but never shrinks (pass the previous range), and `follow` tracks a window of the latest data. Here a noisy random walk streams in.',
}

const fig = figure({ width: 680, aspect: 1.5 })
const grid = gridSpec(2, 2)
const cellOf = ['r0c0', 'r0c1', 'r1c0', 'r1c1'] as const
const policies: readonly { readonly name: string; readonly x: LimitPolicy; readonly y: LimitPolicy }[] = [
  { name: 'auto', x: { kind: 'auto' }, y: { kind: 'auto' } },
  { name: 'fixed', x: { kind: 'fixed', range: [0, 400] }, y: { kind: 'fixed', range: [-20, 20] } },
  { name: 'expand', x: { kind: 'expand' }, y: { kind: 'expand' } },
  { name: 'follow', x: { kind: 'follow', window: 80 }, y: { kind: 'auto' } },
]

// A deterministic walk: a sum of slow sines stands in for noise.
const walk = Array.from(
  { length: 400 },
  (_, i) => 8 * Math.sin(i / 23) + 4 * Math.sin(i / 7.3) + 2 * Math.sin(i / 2.1) + i / 40,
)

export default function LimitPolicies(): ReactNode {
  const mode = useColourMode()
  const [n, setN] = useState(20)
  const [running, setRunning] = useState(true)
  const [speed, setSpeed] = useState(2)
  const previous = useRef<Record<string, { x?: Range; y?: Range }>>({})
  useEffect(() => {
    if (!running) return
    const id = setInterval(() => setN((k) => (k >= walk.length ? 20 : k + speed)), 50)
    return () => clearInterval(id)
  }, [running, speed])
  // One figure with four panels, each with its own limit policy; the ranges each panel showed feed `expand`.
  const option = useMemo(() => {
    if (n === 20) previous.current = {}
    const panels = Object.fromEntries(
      policies.map((p, i) => {
        const prev = previous.current[p.name]
        return [
          cellOf[i] ?? 'r0c0',
          {
            layers: [lines([{ y: walk.slice(0, n) }], { mode })],
            options: {
              title: p.name,
              x: { limits: p.x },
              y: { limits: p.y },
              ...(prev === undefined ? {} : { previous: prev }),
            },
          },
        ]
      }),
    )
    const opt = subplots(grid, panels, { mode, figure: fig })
    const axes = opt as { xAxis: { min: number; max: number }[]; yAxis: { min: number; max: number }[] }
    policies.forEach((p, i) => {
      previous.current[p.name] = {
        x: [axes.xAxis[i]?.min ?? 0, axes.xAxis[i]?.max ?? 1],
        y: [axes.yAxis[i]?.min ?? 0, axes.yAxis[i]?.max ?? 1],
      }
    })
    return opt
  }, [mode, n])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Toggle label="streaming" value={running} onChange={setRunning} />
        <NumberField label="points per tick" value={speed} onChange={setSpeed} min={1} max={10} integer />
      </ControlPanel>
      <FigureFrame figure={fig} filename="limit-policies">
        <Chart option={option} />
      </FigureFrame>
    </div>
  )
}
