import { type ReactNode, useMemo } from 'react'
import { boxPlot, cartesian, qqPlot, violinPlot } from 'whitebox-render/charts'
import { figure } from 'whitebox-render/figure'
import { Chart, FigureFrame, useColourMode } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Box, violin and QQ plots',
  question: 'How do I summarise samples and check them against a normal distribution?',
  explain:
    '`boxPlot` draws Tukey boxes (quartiles, whiskers at 1.5 IQR, outliers), `violinPlot` mirrored kernel densities with quartiles, and `qqPlot` sorted data against normal quantiles with the line through the quartiles.',
}

const fig = figure({ width: 300, aspect: 1 })
const skewed = Array.from({ length: 120 }, (_, i) => Math.exp(1.2 * Math.sin(i * 2.39996) * Math.sqrt((i + 1) / 120)))
const groups = [
  { name: 'A', values: Array.from({ length: 80 }, (_, i) => Math.sin(i * 2.39996) * 1.2 + 2) },
  { name: 'B', values: skewed },
  { name: 'C', values: [...Array.from({ length: 60 }, (_, i) => 3 + Math.cos(i * 1.7)), 7.5] },
]

export default function BoxViolinQQ(): ReactNode {
  const mode = useColourMode()
  const options = useMemo(
    () => [
      { key: 'box', option: cartesian([boxPlot(groups, { mode })], { mode, figure: fig, title: 'box' }) },
      { key: 'violin', option: cartesian([violinPlot(groups, { mode })], { mode, figure: fig, title: 'violin' }) },
      {
        key: 'qq',
        option: cartesian([qqPlot(skewed, { mode })], {
          mode,
          figure: fig,
          title: 'QQ (B)',
          x: { label: 'normal quantile' },
        }),
      },
    ],
    [mode],
  )
  return (
    <div className="flex flex-wrap gap-3">
      {options.map((o) => (
        <FigureFrame key={o.key} figure={fig} exportFormats={['png']}>
          <Chart option={o.option} />
        </FigureFrame>
      ))}
    </div>
  )
}
