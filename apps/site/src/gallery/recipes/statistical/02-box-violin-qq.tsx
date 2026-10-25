import { key, normal } from 'mdml-engine/random'
import { boxPlot, qqPlot, subplots, violinPlot } from 'mdml-render/charts'
import { figure, gridSpec } from 'mdml-render/figure'
import { Chart, ControlPanel, FigureFrame, NumberField, useColourMode } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Box, violin and QQ plots',
  question: 'How do I summarise samples and check them against a normal distribution?',
  explain:
    '`boxPlot` draws Tukey boxes (quartiles, whiskers at 1.5 IQR, outliers), `violinPlot` mirrored kernel densities with quartiles, and `qqPlot` sorted data against normal quantiles with the line through the quartiles. Raise the skew of sample B and watch its QQ plot bend.',
}

const fig = figure({ width: 860, aspect: 2.6 })
const grid = gridSpec(1, 3)
const z = Array.from(normal(key(5), [300]).data)

export default function BoxViolinQQ(): ReactNode {
  const mode = useColourMode()
  const [skew, setSkew] = useState(0.8)
  const option = useMemo(() => {
    const a = z.slice(0, 100).map((v) => 2 + v)
    // A log-normal: exp(skew * z), standardised to a similar spread.
    const b = z.slice(100, 200).map((v) => (skew === 0 ? v : (Math.exp(skew * v) - 1) / skew) + 3)
    const c = [...z.slice(200, 260).map((v) => 3 + 0.5 * v), 7.5]
    const groups = [
      { name: 'A', values: a },
      { name: 'B', values: b },
      { name: 'C', values: c },
    ]
    return subplots(
      grid,
      {
        r0c0: { layers: [boxPlot(groups, { mode })], options: { title: 'box' } },
        r0c1: { layers: [violinPlot(groups, { mode })], options: { title: 'violin' } },
        r0c2: { layers: [qqPlot(b, { mode })], options: { title: 'QQ of B', x: { label: 'normal quantile' } } },
      },
      { mode, figure: fig },
    )
  }, [mode, skew])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={1}>
        <NumberField label="skew of B" value={skew} onChange={setSkew} min={0} max={1.5} />
      </ControlPanel>
      <FigureFrame figure={fig} filename="box-violin-qq">
        <Chart option={option} />
      </FigureFrame>
    </div>
  )
}
