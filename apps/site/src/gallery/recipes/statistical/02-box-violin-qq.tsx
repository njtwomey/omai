import { type ReactNode, useMemo, useState } from 'react'
import { key, normal } from 'whitebox-engine/random'
import { boxPlot, cartesian, qqPlot, violinPlot } from 'whitebox-render/charts'
import { figure } from 'whitebox-render/figure'
import { Chart, ControlPanel, FigureFrame, NumberField, useColourMode } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Box, violin and QQ plots',
  question: 'How do I summarise samples and check them against a normal distribution?',
  explain:
    '`boxPlot` draws Tukey boxes (quartiles, whiskers at 1.5 IQR, outliers), `violinPlot` mirrored kernel densities with quartiles, and `qqPlot` sorted data against normal quantiles with the line through the quartiles. Raise the skew of sample B and watch its QQ plot bend.',
}

const fig = figure({ width: 300, aspect: 1 })
const z = Array.from(normal(key(5), [300]).data)

export default function BoxViolinQQ(): ReactNode {
  const mode = useColourMode()
  const [skew, setSkew] = useState(0.8)
  const options = useMemo(() => {
    const a = z.slice(0, 100).map((v) => 2 + v)
    // A log-normal: exp(skew * z), standardised to a similar spread.
    const b = z.slice(100, 200).map((v) => (skew === 0 ? v : (Math.exp(skew * v) - 1) / skew) + 3)
    const c = [...z.slice(200, 260).map((v) => 3 + 0.5 * v), 7.5]
    const groups = [
      { name: 'A', values: a },
      { name: 'B', values: b },
      { name: 'C', values: c },
    ]
    return [
      { key: 'box', option: cartesian([boxPlot(groups, { mode })], { mode, figure: fig, title: 'box' }) },
      { key: 'violin', option: cartesian([violinPlot(groups, { mode })], { mode, figure: fig, title: 'violin' }) },
      {
        key: 'qq',
        option: cartesian([qqPlot(b, { mode })], {
          mode,
          figure: fig,
          title: 'QQ of B',
          x: { label: 'normal quantile' },
        }),
      },
    ]
  }, [mode, skew])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={1}>
        <NumberField label="skew of B" value={skew} onChange={setSkew} min={0} max={1.5} />
      </ControlPanel>
      <div className="flex flex-wrap gap-3">
        {options.map((o) => (
          <FigureFrame key={o.key} figure={fig} exportFormats={['png']}>
            <Chart option={o.option} />
          </FigureFrame>
        ))}
      </div>
    </div>
  )
}
