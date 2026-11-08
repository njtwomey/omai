import { cartesian, maskMatrix } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import { Chart, Choice, ControlPanel, FigureFrame, NumberField, useColourMode } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Mask matrix',
  question: 'How do I show which cells of a boolean matrix are on, such as an attention mask?',
  explain:
    '`maskMatrix(mask, { on, off, rowLabels, colLabels, rowColours, colColours, marks })` draws a boolean matrix as two colours, with `rowColours` and `colColours` tinting the labels (here the padding tokens of a sentence) and `marks` writing a symbol in a cell. The causal mask lets each token see the ones before it; the window mask only the nearest few.',
}

const FIGURE = figure({ width: 640, height: 400 })
const TOKENS = ['the', 'cat', 'sat', 'on', 'the', 'mat', '<pad>', '<pad>']

export default function MaskMatrixRecipe(): ReactNode {
  const mode = useColourMode()
  const [kind, setKind] = useState<'causal' | 'window' | 'full'>('causal')
  const [window, setWindow] = useState(3)
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
          rowLabels: TOKENS,
          colLabels: TOKENS,
          rowColours: tint,
          colColours: tint,
          marks: (on, i, j) => (on && i === j ? '✓' : undefined),
        }),
      ],
      { mode, figure: FIGURE, x: { label: 'attended to' }, y: { label: 'query' } },
    )
  }, [kind, window, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Choice label="mask" value={kind} options={['causal', 'window', 'full'] as const} onChange={setKind} />
        <NumberField label="window" value={window} onChange={setWindow} min={1} max={8} integer />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="mask-matrix">
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
    </div>
  )
}
