import { cartesian, maskMatrix } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import { Chart, Choice, ControlPanel, FigureFrame, NumberField, useColourMode } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Mask matrix',
  question: 'How do I show which cells of a boolean matrix are on, such as an attention mask?',
  explain:
    '`maskMatrix(mask, { on, off, rowLabels, colLabels, rowColours, colColours, marks })` draws a boolean matrix as two colours, with `rowColours` and `colColours` tinting the labels (here the padding tokens of a sentence) and `marks` writing a symbol in a cell. `on` may be a function of the row and column giving each cell its colour, here by the part of the sentence attended to (the first and last words and the rest). `ticks` sets the font family and weight of the tick labels. The causal mask lets each token see the ones before it; the window mask only the nearest few.',
}

const FIGURE = figure({ width: 640, height: 400 })
const TOKENS = ['the', 'cat', 'sat', 'on', 'the', 'mat', '<pad>', '<pad>']

export default function MaskMatrixRecipe(): ReactNode {
  const mode = useColourMode()
  const [kind, setKind] = useState<'causal' | 'window' | 'full'>('causal')
  const [window, setWindow] = useState(3)
  const [byPart, setByPart] = useState<'one colour' | 'by part'>('by part')
  const option = useMemo(() => {
    const real = TOKENS.map((t) => t !== '<pad>')
    const allowed = (i: number, j: number): boolean =>
      real[j] === true && (kind === 'full' || (j <= i && (kind === 'causal' || i - j < window)))
    const mask = TOKENS.map((_, i) => TOKENS.map((__, j) => allowed(i, j)))
    const tint = TOKENS.map((t) => (t === '<pad>' ? 'tab1' : undefined))
    return cartesian(
      [
        maskMatrix(mask, {
          mode,
          ...(byPart === 'by part'
            ? { on: (_row: number, col: number): string => (col === 0 ? 'tab2' : col === 5 ? 'tab3' : 'tab0') }
            : {}),
          rowLabels: TOKENS,
          colLabels: TOKENS,
          rowColours: tint,
          colColours: tint,
          marks: (on, i, j) => (on && i === j ? '✓' : undefined),
        }),
      ],
      {
        mode,
        figure: FIGURE,
        x: { label: 'attended to', ticks: { fontFamily: 'ui-monospace, monospace' } },
        y: { label: 'query', ticks: { fontFamily: 'ui-monospace, monospace' } },
      },
    )
  }, [kind, window, byPart, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <Choice label="mask" value={kind} options={['causal', 'window', 'full'] as const} onChange={setKind} />
        <Choice label="cells" value={byPart} options={['one colour', 'by part'] as const} onChange={setByPart} />
        <NumberField label="window" value={window} onChange={setWindow} min={1} max={8} integer />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="mask-matrix">
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
    </div>
  )
}
