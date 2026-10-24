import { type ReactNode, useMemo, useState } from 'react'
import { key, normal } from 'whitebox-engine/random'
import { toArray } from 'whitebox-engine/tensor'
import {
  Brush,
  type DataRect,
  InteractionOverlay,
  Overlaid,
  type Point2,
  Probe,
  pointsInRect,
} from 'whitebox-render/react'
import { Dots, Plot, plotFrame } from '../../demo/Plot.tsx'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Probe and brush',
  question: 'How do I read values under the pointer, and select a region of the data?',
  explain:
    '`Probe` follows the pointer with a crosshair and hands its data coordinates to whatever it draws; `Brush` selects a rectangle by dragging (a press clears it) and `pointsInRect` finds the data inside. Both sit in the same kind of overlay as the draggable points.',
}

const frame = plotFrame({ x: [-3, 3], y: [-3, 3], width: 420, aspect: 1.2 })

export default function ProbeAndBrush(): ReactNode {
  const points = useMemo(() => {
    const xy = toArray(normal(key(7), [150, 2])) as number[][]
    return xy.map(([a = 0, b = 0]): Point2 => [a, 0.6 * a + 0.8 * b])
  }, [])
  const [rect, setRect] = useState<DataRect | null>(null)
  const selected = new Set(rect === null ? [] : pointsInRect(points, rect))
  const size = { width: frame.fig.width, height: frame.fig.height }
  return (
    <div className="flex flex-wrap gap-4">
      <div className="flex flex-col gap-1">
        <span className="text-xs text-muted">Probe</span>
        <Overlaid
          {...size}
          overlay={
            <InteractionOverlay projection={frame.projection} {...size}>
              <Probe>
                {([x, y], [px, py]) => (
                  <text x={px + 8} y={py - 8} fontSize={11} fill="var(--wb-fg)">
                    ({x.toFixed(2)}, {y.toFixed(2)})
                  </text>
                )}
              </Probe>
            </InteractionOverlay>
          }
        >
          <Plot frame={frame}>
            <Dots frame={frame} points={points} />
          </Plot>
        </Overlaid>
      </div>
      <div className="flex flex-col gap-1">
        <span className="text-xs text-muted">
          Brush: {selected.size} of {points.length} selected
        </span>
        <Overlaid
          {...size}
          overlay={
            <InteractionOverlay projection={frame.projection} {...size}>
              <Brush value={rect} onChange={setRect} />
            </InteractionOverlay>
          }
        >
          <Plot frame={frame}>
            <Dots
              frame={frame}
              points={points}
              colour={(i) => (rect === null || selected.has(i) ? 'var(--wb-tab0)' : 'var(--wb-muted)')}
            />
          </Plot>
        </Overlaid>
      </div>
    </div>
  )
}
