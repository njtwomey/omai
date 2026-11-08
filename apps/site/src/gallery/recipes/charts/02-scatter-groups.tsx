import { key, normal } from 'omai-engine/random'
import { cartesian, points } from 'omai-render/charts'
import { Chart, ControlPanel, FigureFrame, NumberField, Toggle, useColourMode } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Scatter plots',
  question: 'How do I scatter points by class, with sizes and marker shapes?',
  explain:
    '`scatterChart` (or the `points` layer) makes one series per group in tab10 order. `shapes: true` varies the marker by group so classes never rely on colour alone, and `sizes` maps values to marker area. A point set whose `symbol` is none draws only its `labels`, sized by `labelSize`, placed by `labelPosition` and coloured by `labelColour`: here the class names written over their clusters. Noise comes from a seeded key, so the picture is the same on every visit.',
}

const n = 40
const noise = Array.from(normal(key(7), [3 * n * 2]).data)

export default function ScatterGroups(): ReactNode {
  const mode = useColourMode()
  const [spread, setSpread] = useState(0.5)
  const [shapes, setShapes] = useState(true)
  const [bubbles, setBubbles] = useState(false)
  const [names, setNames] = useState(true)
  const option = useMemo(() => {
    const pts = [0, 1, 2].flatMap((g) =>
      Array.from({ length: n }, (_, i) => {
        const a = (2 * Math.PI * g) / 3
        const k = 2 * (g * n + i)
        return { g, x: 2 * Math.cos(a) + spread * (noise[k] ?? 0), y: 2 * Math.sin(a) + spread * (noise[k + 1] ?? 0) }
      }),
    )
    const centres = [0, 1, 2].map((g) => (2 * Math.PI * g) / 3)
    return cartesian(
      [
        points(
          [
            {
              x: pts.map((p) => p.x),
              y: pts.map((p) => p.y),
              groups: pts.map((p) => p.g),
              groupNames: { 0: 'class a', 1: 'class b', 2: 'class c' },
              ...(bubbles ? { sizes: pts.map((p) => Math.hypot(p.x, p.y)), sizeRange: [4, 16] as const } : {}),
              shapes,
            },
          ],
          { mode },
        ),
        ...(names
          ? [
              points(
                [
                  {
                    x: centres.map((a) => 3.4 * Math.cos(a)),
                    y: centres.map((a) => 3.4 * Math.sin(a)),
                    labels: ['class a', 'class b', 'class c'],
                    symbol: 'none',
                    labelSize: 14,
                    labelPosition: 'inside',
                    labelColour: 'fg',
                  },
                ],
                { mode },
              ),
            ]
          : []),
      ],
      { mode, aspect: 'equal', x: { label: 'x₁', limits: [-4.5, 4.5] }, y: { label: 'x₂', limits: [-4, 4] } },
    )
  }, [mode, spread, shapes, bubbles, names])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={4}>
        <NumberField label="spread" value={spread} onChange={setSpread} min={0.05} max={2} />
        <Toggle label="marker shapes" value={shapes} onChange={setShapes} />
        <Toggle label="size by radius" value={bubbles} onChange={setBubbles} />
        <Toggle label="class names" value={names} onChange={setNames} />
      </ControlPanel>
      <FigureFrame filename="scatter">
        <Chart option={option} />
      </FigureFrame>
    </div>
  )
}
