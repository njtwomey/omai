import { type ReactNode, useEffect, useMemo, useRef, useState } from 'react'
import { cartesian, lines } from 'whitebox-render/charts'
import { figure, type LimitPolicy, type Range } from 'whitebox-render/figure'
import { Chart, FigureFrame, useColourMode } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Limit policies',
  question: 'How should an axis range behave while the data change?',
  explain:
    'A reactive figure picks a `LimitPolicy` per axis: `auto` fits the data, `fixed` holds a stated range (so movement is visible), `expand` grows but never shrinks (pass the previous range), and `follow` tracks a window of the latest data. Here a noisy random walk streams in.',
}

const fig = figure({ width: 300, aspect: 1.2 })
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
  const previous = useRef<Record<string, { x?: Range; y?: Range }>>({})
  useEffect(() => {
    const id = setInterval(() => setN((k) => (k >= walk.length ? 20 : k + 2)), 50)
    return () => clearInterval(id)
  }, [])
  const options = useMemo(
    () =>
      policies.map((p) => {
        if (n === 20) previous.current = {}
        const prev = previous.current[p.name]
        const opt = cartesian([lines([{ y: walk.slice(0, n) }], { mode })], {
          mode,
          figure: fig,
          title: p.name,
          x: { limits: p.x },
          y: { limits: p.y },
          ...(prev === undefined ? {} : { previous: prev }),
        })
        const axes = opt as { xAxis: { min: number; max: number }[]; yAxis: { min: number; max: number }[] }
        previous.current[p.name] = {
          x: [axes.xAxis[0]?.min ?? 0, axes.xAxis[0]?.max ?? 1],
          y: [axes.yAxis[0]?.min ?? 0, axes.yAxis[0]?.max ?? 1],
        }
        return opt
      }),
    [mode, n],
  )
  return (
    <div className="flex flex-wrap gap-3">
      {options.map((o, i) => (
        <FigureFrame key={policies[i]?.name} figure={fig} exportFormats={[]}>
          <Chart option={o} />
        </FigureFrame>
      ))}
    </div>
  )
}
