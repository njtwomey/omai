import { type ReactNode, useMemo, useState } from 'react'
import { normal, studentT } from 'whitebox-engine/dists'
import { densityChart } from 'whitebox-render/charts'
import { figure } from 'whitebox-render/figure'
import { Chart, ControlPanel, FigureFrame, NumberField, useColourMode } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Densities',
  question: 'How do I plot the densities of a few distributions together?',
  explain:
    '`densityChart([{ name, dist }])` draws each whitebox-engine `Distribution` over a range that covers them all (each one’s 0.1% to 99.9% quantiles unless `range` is given). Lower the degrees of freedom to watch the Student t grow its heavy tails against the normal.',
}

// The fixed figure the chart fills (the builders' default size).
const FIGURE = figure()

export default function Densities(): ReactNode {
  const mode = useColourMode()
  const [df, setDf] = useState(3)
  const option = useMemo(
    () =>
      densityChart(
        [
          { name: 'normal', dist: normal(0, 1) },
          { name: `Student t (ν = ${df})`, dist: studentT(df) },
        ],
        { mode, range: [-6, 6], x: { label: 'x' }, y: { label: 'density' } },
      ),
    [df, mode],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel>
        <NumberField label="degrees of freedom ν" value={df} onChange={setDf} min={0.5} max={100} log />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="densities">
        <Chart option={option} />
      </FigureFrame>
    </div>
  )
}
