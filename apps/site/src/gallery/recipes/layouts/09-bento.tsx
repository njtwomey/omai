import { type ReactNode, useMemo } from 'react'
import { heatmap, histogram, lines, type Panel, points } from 'whitebox-render/charts'
import { bento, figure } from 'whitebox-render/figure'
import { FigureFrame, FigureGrid, useColourMode } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Bento layouts',
  question: 'How do I lay out a dashboard of panels by naming areas?',
  explain:
    "`bento(['a a b', 'c d b'])` reads rows of names like CSS `grid-template-areas`: a name repeated over a rectangle spans it, and `.` leaves a hole. Each named cell gets a panel; `kinds` makes a cell text or TeX.",
}

const fig = figure({ width: 720, aspect: 1.5 })
const grid = bento(['loss loss field', 'scatter hist field', 'scatter hist note'], {
  wspace: 56,
  hspace: 56,
  kinds: { note: 'text' },
})
const steps = Array.from({ length: 200 }, (_, i) => i)
const z = Array.from({ length: 20 }, (_, i) => Array.from({ length: 20 }, (_, j) => Math.sin(i / 3) * Math.cos(j / 4)))

export default function Bento(): ReactNode {
  const mode = useColourMode()
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
    <FigureFrame figure={fig}>
      <FigureGrid
        figure={fig}
        grid={grid}
        panels={panels}
        cells={{ note: <span className="text-sm text-[var(--wb-muted)]">a text cell in the bento</span> }}
      />
    </FigureFrame>
  )
}
