import { type ReactNode, useMemo, useState } from 'react'
import { lines, type Panel } from 'whitebox-render/charts'
import { figure, gridSpec } from 'whitebox-render/figure'
import { ControlPanel, FigureFrame, FigureGrid, Toggle, useColourMode } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Shared axes',
  question: 'How do panels share an x or y range?',
  explain:
    "A cell with `sharex: 'id'` (or `sharey`) uses that cell’s axis range: the union of their data under the first panel’s limits, with inner tick labels hidden, and hover linked across them. Turn sharing off to see each panel fit its own data.",
}

const fig = figure({ width: 640, aspect: 1.3 })
const curve = (scale: number, shift: number, f: (x: number) => number): { x: number[]; y: number[] } => {
  const x = Array.from({ length: 100 }, (_, i) => shift + (i / 99) * scale)
  return { x, y: x.map(f) }
}

export default function SharedAxes(): ReactNode {
  const mode = useColourMode()
  const [sharex, setSharex] = useState(true)
  const [sharey, setSharey] = useState(true)
  const grid = useMemo(
    () =>
      gridSpec(2, 2, {
        cells: [
          { id: 'a', row: 0, col: 0 },
          { id: 'b', row: 0, col: 1, ...(sharey ? { sharey: 'a' } : {}) },
          { id: 'c', row: 1, col: 0, ...(sharex ? { sharex: 'a' } : {}) },
          { id: 'd', row: 1, col: 1, ...(sharex ? { sharex: 'b' } : {}), ...(sharey ? { sharey: 'c' } : {}) },
        ],
      }),
    [sharex, sharey],
  )
  const panels = useMemo<Record<string, Panel>>(
    () => ({
      a: { layers: [lines([curve(6, 0, Math.sin)], { mode })] },
      b: { layers: [lines([curve(10, 2, (x) => 2 * Math.cos(x))], { mode, colourOffset: 1 })] },
      c: { layers: [lines([curve(4, 1, (x) => x / 3)], { mode, colourOffset: 2 })] },
      d: { layers: [lines([curve(8, -2, (x) => 0.5 * Math.sin(2 * x))], { mode, colourOffset: 3 })] },
    }),
    [mode],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Toggle label="share x (down columns)" value={sharex} onChange={setSharex} />
        <Toggle label="share y (across rows)" value={sharey} onChange={setSharey} />
      </ControlPanel>
      <FigureFrame figure={fig}>
        <FigureGrid figure={fig} grid={grid} panels={panels} tooltip="axis" />
      </FigureFrame>
    </div>
  )
}
