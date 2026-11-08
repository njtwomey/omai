import { bars, histogram, lines, type Panel, points } from 'omai-render/charts'
import { figure, gridSpec } from 'omai-render/figure'
import { ControlPanel, FigureFrame, FigureGrid, NumberField, useColourMode } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Grids with spans and ratios',
  question: 'How do I lay out several axes on a grid, some spanning rows or columns?',
  explain:
    '`gridSpec(rows, cols, { widthRatios, heightRatios, wspace, hspace, cells })` is matplotlib’s gridspec: each cell names its row, column and spans. `FigureGrid` (or `subplots` for a bare ECharts option) draws a panel of layers in each cell.',
}

const fig = figure({ width: 680, aspect: 1.45 })
const t = Array.from({ length: 120 }, (_, i) => i / 10)
const noise = t.map((v, i) => Math.sin(v * 3.1 + i) * 0.3)

export default function GridSpecRecipe(): ReactNode {
  const mode = useColourMode()
  const [ratio, setRatio] = useState(2)
  const [gap, setGap] = useState(40)
  const grid = useMemo(
    () =>
      gridSpec(2, 3, {
        widthRatios: [ratio, 1, 1],
        heightRatios: [1, 1],
        wspace: gap,
        hspace: gap,
        cells: [
          { id: 'wide', row: 0, col: 0, colSpan: 2 },
          { id: 'tall', row: 0, col: 2, rowSpan: 2 },
          { id: 'a', row: 1, col: 0 },
          { id: 'b', row: 1, col: 1 },
        ],
      }),
    [ratio, gap],
  )
  const panels = useMemo<Record<string, Panel>>(
    () => ({
      wide: {
        layers: [lines([{ x: t, y: t.map(Math.sin), name: 'signal' }], { mode })],
        options: { title: 'wide: colSpan 2' },
      },
      tall: { layers: [histogram(noise, { bins: 12, horizontal: true, mode })], options: { title: 'tall: rowSpan 2' } },
      a: {
        layers: [points([{ x: t.map(Math.sin), y: t.map(Math.cos), size: 4 }], { mode })],
        options: { aspect: 'equal' },
      },
      b: { layers: [bars(['x', 'y', 'z'], [3, 1, 2], { mode })] },
    }),
    [mode],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="first column ratio" value={ratio} onChange={setRatio} min={0.5} max={4} />
        <NumberField label="gap (px)" value={gap} onChange={setGap} min={0} max={80} integer />
      </ControlPanel>
      <FigureFrame figure={fig}>
        <FigureGrid figure={fig} grid={grid} panels={panels} legend={false} />
      </FigureFrame>
    </div>
  )
}
