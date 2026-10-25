import { band, lines, type Panel } from 'mdml-render/charts'
import { figure, gridSpec } from 'mdml-render/figure'
import { ControlPanel, FigureFrame, FigureGrid, NumberField, useColourMode } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Zero-gap panels',
  question: 'How do I stack panels on one time axis with no space between them?',
  explain:
    'Gaps between rows and columns are pixels, down to zero (`hspace: 0`). With a shared x axis only the bottom panel keeps its tick labels, so stacked traces read as one figure: here a signal, its envelope and its phase.',
}

const fig = figure({ width: 640, aspect: 1.2 })
const t = Array.from({ length: 400 }, (_, i) => i / 40)
const env = t.map((v) => 1 + 0.5 * Math.sin(v / 2))
const sig = t.map((v, i) => (env[i] ?? 1) * Math.sin(3 * v))

export default function ZeroGap(): ReactNode {
  const mode = useColourMode()
  const [gap, setGap] = useState(0)
  const grid = useMemo(
    () =>
      gridSpec(3, 1, {
        hspace: gap,
        heightRatios: [2, 1, 1],
        cells: [
          { id: 'signal', row: 0, col: 0 },
          { id: 'envelope', row: 1, col: 0, sharex: 'signal' },
          { id: 'phase', row: 2, col: 0, sharex: 'signal' },
        ],
      }),
    [gap],
  )
  const panels = useMemo<Record<string, Panel>>(
    () => ({
      signal: { layers: [lines([{ x: t, y: sig }], { mode })], options: { y: { label: 'signal' } } },
      envelope: {
        layers: [band([{ x: t, lower: env.map((e) => -e), upper: env }], { mode, colourOffset: 1 })],
        options: { y: { label: 'envelope' } },
      },
      phase: {
        layers: [lines([{ x: t, y: t.map((v) => ((3 * v) % (2 * Math.PI)) - Math.PI) }], { mode, colourOffset: 2 })],
        options: { y: { label: 'phase' }, x: { label: 'time (s)' } },
      },
    }),
    [mode],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={1}>
        <NumberField label="gap between rows (px)" value={gap} onChange={setGap} min={0} max={40} integer />
      </ControlPanel>
      <FigureFrame figure={fig}>
        <FigureGrid figure={fig} grid={grid} panels={panels} tooltip="axis" />
      </FigureFrame>
    </div>
  )
}
