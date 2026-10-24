import { type ReactNode, useMemo, useState } from 'react'
import { normal } from 'whitebox-engine/dists'
import { density, type Panel } from 'whitebox-render/charts'
import { figure, gridSpec } from 'whitebox-render/figure'
import { live, tex, toTex } from 'whitebox-render/maths'
import { ControlPanel, FigureFrame, FigureGrid, NumberField, useColourMode } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Text and equation cells',
  question: 'How do I put an equation or a note in a grid cell beside the axes?',
  explain:
    'Grid cells may hold `text` or `tex` instead of axes. `FigureGrid` typesets `tex` cells with KaTeX and the shared macros, and the `tex` tag binds live values, so the equation changes with the controls.',
}

const fig = figure({ width: 680, aspect: 2 })

export default function TextCells(): ReactNode {
  const mode = useColourMode()
  const [mu, setMu] = useState(0.5)
  const [sigma, setSigma] = useState(1.2)
  const grid = useMemo(
    () =>
      gridSpec(2, 2, {
        widthRatios: [1.4, 1],
        cells: [
          { id: 'plot', row: 0, col: 0, rowSpan: 2 },
          {
            id: 'equation',
            row: 0,
            col: 1,
            kind: 'tex',
            content: toTex(
              tex`p(x) = \frac{1}{\sqrt{2\pi}\,${live(sigma, { digits: 3 })}} \exp\!\left(-\frac{(x - ${live(mu, { digits: 3 })})^2}{2 \cdot ${live(sigma, { digits: 3 })}^2}\right)`,
            ),
          },
          {
            id: 'note',
            row: 1,
            col: 1,
            kind: 'text',
            content: `A normal density, mean ${mu.toFixed(2)}, variance ${(sigma * sigma).toFixed(2)}.`,
          },
        ],
      }),
    [mu, sigma],
  )
  const panels = useMemo<Record<string, Panel>>(
    () => ({
      plot: {
        layers: [density(normal(mu, sigma), { mode, fill: 0.25, range: [-6, 6] })],
        options: { x: { limits: [-6, 6], label: 'x' }, y: { limits: [0, 1.4] } },
      },
    }),
    [mu, sigma, mode],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="mean μ" value={mu} onChange={setMu} min={-3} max={3} />
        <NumberField label="std σ" value={sigma} onChange={setSigma} min={0.3} max={3} />
      </ControlPanel>
      <FigureFrame figure={fig}>
        <FigureGrid figure={fig} grid={grid} panels={panels} />
      </FigureFrame>
    </div>
  )
}
