import { band, cartesian, lines } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import {
  Chart,
  ControlPanel,
  DraggableLine,
  FigureFrame,
  InteractionOverlay,
  NumberField,
  Overlaid,
  useChartProjection,
  useColourMode,
} from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Draggable lines',
  question: 'How do I let the reader drag a threshold or a level on the figure, in step with its control?',
  explain:
    '`DraggableLine` draws a vertical (`axis="x"`) or horizontal (`axis="y"`) line inside an `InteractionOverlay` and calls `onChange` as the reader drags it. Bind its `value` and `onChange` to the same state as a `NumberField`, and the line and the control move together whichever one is used. It snaps to `step`, stays within `min` and `max`, and takes the arrow keys once focused (Shift for ten steps). Give the chart a fixed figure and fixed limits so nothing moves under the hand. `handle` sets the end handle: a `bar` (the default), a `square` or a `circle`, hollow or `solid`, and `DraggablePoints` takes `shape` too. Here: drag x to shade P(X ≤ x) under the normal density, and the level to mark where the density is at least that high.',
}

const FIGURE = figure({ width: 640, height: 360 })
const pdf = (x: number): number => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI)
const xs = Array.from({ length: 241 }, (_, i) => -4 + (8 * i) / 240)
// P(X ≤ x) by the trapezium rule on the grid, close enough for a readout.
const cdf = (x: number): number => {
  let s = 0
  for (let i = 1; i < xs.length && (xs[i] ?? 0) <= x; i++)
    s += ((pdf(xs[i - 1] ?? 0) + pdf(xs[i] ?? 0)) / 2) * (8 / 240)
  return Math.min(1, s)
}

export default function DraggableLines(): ReactNode {
  const mode = useColourMode()
  const { projection, onInit } = useChartProjection()
  const [x, setX] = useState(0.8)
  const [level, setLevel] = useState(0.15)
  // Where the density reaches the level: |t| ≤ sqrt(-2 log(level √(2π))).
  const half = level < pdf(0) ? Math.sqrt(-2 * Math.log(level * Math.sqrt(2 * Math.PI))) : 0
  const option = useMemo(() => {
    const left = xs.filter((t) => t <= x)
    const above = xs.filter((t) => Math.abs(t) <= half)
    return cartesian(
      [
        band([{ x: left, lower: left.map(() => 0), upper: left.map(pdf), name: 'P(X ≤ x)', opacity: 0.3 }], { mode }),
        lines([{ x: xs, y: xs.map(pdf), name: 'N(0, 1) density' }], { mode }),
        lines([{ x: above, y: above.map(() => level), name: 'density ≥ level', width: 4 }], { mode, colourOffset: 2 }),
      ],
      { mode, figure: FIGURE, x: { label: 'x', limits: [-4, 4] }, y: { label: 'density', limits: [0, 0.45] } },
    )
  }, [x, level, half, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="x" value={x} onChange={setX} min={-4} max={4} step={0.05} />
        <NumberField label="level" value={level} onChange={setLevel} min={0} max={0.4} step={0.005} />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="draggable-lines">
        <Overlaid
          width={FIGURE.width}
          height={FIGURE.height}
          overlay={
            projection === null ? null : (
              <InteractionOverlay projection={projection} width={FIGURE.width} height={FIGURE.height}>
                <DraggableLine
                  axis="x"
                  value={x}
                  onChange={setX}
                  span={[0, 0.45]}
                  min={-4}
                  max={4}
                  step={0.05}
                  label={`x = ${x.toFixed(2)} · P(X ≤ x) = ${cdf(x).toFixed(3)}`}
                />
                <DraggableLine
                  axis="y"
                  value={level}
                  onChange={setLevel}
                  span={[-4, 4]}
                  min={0}
                  max={0.4}
                  step={0.005}
                  colour="var(--wb-tab2)"
                  handle={{ shape: 'square', size: 9, fill: 'solid' }}
                  label={`level ${level.toFixed(3)}`}
                />
              </InteractionOverlay>
            )
          }
        >
          <Chart option={option} figure={FIGURE} onInit={onInit} />
        </Overlaid>
      </FigureFrame>
    </div>
  )
}
