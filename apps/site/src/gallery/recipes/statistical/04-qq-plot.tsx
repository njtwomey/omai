import { type ReactNode, useMemo, useState } from 'react'
import { studentT } from 'whitebox-engine/dists'
import { key } from 'whitebox-engine/random'
import { toArray } from 'whitebox-engine/tensor'
import { cartesian, qqPlot } from 'whitebox-render/charts'
import { figure } from 'whitebox-render/figure'
import { Chart, ControlPanel, NumberField, useColourMode } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'QQ plot',
  question: 'How do I check whether a sample looks normal?',
  explain:
    '`qqPlot(samples, { reference })` plots the sorted sample against the reference’s quantiles (standard normal by default) with the line through the quartiles. A Student t sample bends away from the line at both ends as its degrees of freedom fall.',
}

// The fixed figure the chart fills (the builders' default size).
const FIGURE = figure()

export default function QQ(): ReactNode {
  const mode = useColourMode()
  const [df, setDf] = useState(3)
  const option = useMemo(() => {
    const sample = toArray(studentT(df).sample(key(4), [150])) as number[]
    return cartesian([qqPlot(sample, { mode })], {
      mode,
      aspect: 'equal',
      x: { label: 'normal quantiles' },
      y: { label: 'sample' },
    })
  }, [df, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel>
        <NumberField label="degrees of freedom ν" value={df} onChange={setDf} min={1} max={100} log />
      </ControlPanel>
      <Chart option={option} figure={FIGURE} />
    </div>
  )
}
