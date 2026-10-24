import { type ReactNode, useMemo, useState } from 'react'
import { key, normal } from 'whitebox-engine/random'
import { toArray } from 'whitebox-engine/tensor'
import { cartesian, points as pointLayer } from 'whitebox-render/charts'
import { brushSelection, figure } from 'whitebox-render/figure'
import {
  Brush,
  Chart,
  type DataRect,
  InteractionOverlay,
  Overlaid,
  type Point2,
  Probe,
  useChartProjection,
  useColourMode,
} from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Probe and brush',
  question: 'How do I read values under the pointer, and select a region of the data?',
  explain:
    '`Probe` follows the pointer with a crosshair and hands its data coordinates to whatever it draws; `Brush` selects a rectangle by dragging (a press clears it) and `brushSelection` (from `whitebox-render/figure`) finds the data inside. Both sit in the same kind of overlay as the draggable points.',
}

const FIGURE = figure({ width: 440, aspect: 1.2 })
const AXES = { x: { limits: [-3, 3] as const }, y: { limits: [-3, 3] as const }, tooltip: false as const }

export default function ProbeAndBrush(): ReactNode {
  const mode = useColourMode()
  const probe = useChartProjection()
  const brush = useChartProjection()
  const data = useMemo(() => {
    const xy = toArray(normal(key(7), [150, 2])) as number[][]
    return xy.map(([a = 0, b = 0]): Point2 => [a, 0.6 * a + 0.8 * b])
  }, [])
  const [rect, setRect] = useState<DataRect | null>(null)
  const selected = useMemo(() => (rect === null ? null : new Set(brushSelection(data, rect))), [data, rect])
  const plain = useMemo(
    () =>
      cartesian([pointLayer([{ x: data.map((p) => [...p]), name: 'data' }], { mode })], {
        mode,
        figure: FIGURE,
        ...AXES,
      }),
    [data, mode],
  )
  const coloured = useMemo(() => {
    const inside = data.filter((_, i) => selected === null || selected.has(i)).map((p) => [...p])
    const outside = data.filter((_, i) => selected !== null && !selected.has(i)).map((p) => [...p])
    const sets = [
      ...(outside.length > 0 ? [{ x: outside, name: 'outside', colour: 'muted' }] : []),
      ...(inside.length > 0 ? [{ x: inside, name: 'selected' }] : []),
    ]
    return cartesian([pointLayer(sets, { mode })], { mode, figure: FIGURE, legend: false, ...AXES })
  }, [data, selected, mode])
  const size = { width: FIGURE.width, height: FIGURE.height }
  return (
    <div className="flex flex-wrap gap-4">
      <div className="flex flex-col gap-1">
        <span className="text-xs text-muted">Probe</span>
        <Overlaid
          {...size}
          overlay={
            probe.projection === null ? null : (
              <InteractionOverlay projection={probe.projection} {...size}>
                <Probe>
                  {([x, y], [px, py]) => (
                    <text x={px + 8} y={py - 8} fontSize={11} fill="var(--wb-fg)">
                      ({x.toFixed(2)}, {y.toFixed(2)})
                    </text>
                  )}
                </Probe>
              </InteractionOverlay>
            )
          }
        >
          <Chart option={plain} figure={FIGURE} onInit={probe.onInit} />
        </Overlaid>
      </div>
      <div className="flex flex-col gap-1">
        <span className="text-xs text-muted">
          Brush: {selected?.size ?? data.length} of {data.length} selected
        </span>
        <Overlaid
          {...size}
          overlay={
            brush.projection === null ? null : (
              <InteractionOverlay projection={brush.projection} {...size}>
                <Brush value={rect} onChange={setRect} />
              </InteractionOverlay>
            )
          }
        >
          <Chart option={coloured} figure={FIGURE} onInit={brush.onInit} />
        </Overlaid>
      </div>
    </div>
  )
}
