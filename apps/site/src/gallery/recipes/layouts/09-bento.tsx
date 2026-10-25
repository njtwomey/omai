import { heatmap, histogram, lines, type Panel, points } from 'mdml-render/charts'
import { bento, figure } from 'mdml-render/figure'
import { ControlPanel, FigureFrame, FigureGrid, Select, useColourMode } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Bento layouts',
  question: 'How do I lay out a dashboard of panels by naming areas?',
  explain:
    "`bento(['a a b', 'c d b'])` reads rows of names like CSS `grid-template-areas`: a name repeated over a rectangle spans it, and `.` leaves a hole. Each named cell gets a panel; `kinds` makes a cell text or TeX. Pick another arrangement: the panels move, the figure keeps its size.",
}

const fig = figure({ width: 720, aspect: 1.5 })
// Three arrangements of the same five cells, as rows of names.
const LAYOUTS: Readonly<Record<string, readonly string[]>> = {
  'wide loss': ['loss loss field', 'scatter hist field', 'scatter hist note'],
  'tall field': ['field loss loss', 'field scatter hist', 'note scatter hist'],
  strip: ['loss loss loss', 'scatter hist field', 'note note note'],
}
const steps = Array.from({ length: 200 }, (_, i) => i)
const z = Array.from({ length: 20 }, (_, i) => Array.from({ length: 20 }, (_, j) => Math.sin(i / 3) * Math.cos(j / 4)))

export default function Bento(): ReactNode {
  const mode = useColourMode()
  const [layout, setLayout] = useState('wide loss')
  const grid = useMemo(
    () => bento(LAYOUTS[layout] ?? [], { wspace: 56, hspace: 56, kinds: { note: 'text' } }),
    [layout],
  )
  const panels = useMemo<Record<string, Panel>>(
    () => ({
      loss: {
        layers: [
          lines(
            [
              { x: steps, y: steps.map((s) => 2 * Math.exp(-s / 40) + 0.1 + 0.05 * Math.sin(s)), name: 'train' },
              { x: steps, y: steps.map((s) => 2 * Math.exp(-s / 50) + 0.25), name: 'valid' },
            ],
            { mode },
          ),
        ],
        options: { title: 'loss' },
      },
      field: { layers: [heatmap(z, { mode, colourBar: false })], options: { title: 'field' } },
      scatter: {
        layers: [
          points([{ x: steps.map((s) => Math.sin(s * 0.7)), y: steps.map((s) => Math.cos(s * 1.3)), size: 4 }], {
            mode,
          }),
        ],
        options: { title: 'scatter' },
      },
      hist: {
        layers: [
          histogram(
            steps.map((s) => Math.sin(s * 0.37)),
            { bins: 10, mode },
          ),
        ],
        options: { title: 'histogram' },
      },
    }),
    [mode],
  )
  return (
    <div className="flex w-fit max-w-full flex-col gap-3">
      <ControlPanel>
        <Select label="areas" value={layout} options={Object.keys(LAYOUTS)} onChange={setLayout} />
      </ControlPanel>
      <p className="m-0 font-mono text-xs text-muted">{(LAYOUTS[layout] ?? []).map((r) => `'${r}'`).join('  ')}</p>
      <FigureFrame figure={fig} filename="bento">
        <FigureGrid
          figure={fig}
          grid={grid}
          panels={panels}
          cells={{ note: <span className="text-sm text-[var(--wb-muted)]">a text cell in the bento</span> }}
        />
      </FigureFrame>
    </div>
  )
}
