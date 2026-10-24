import { type ReactNode, useMemo, useState } from 'react'
import { gamma, normal, studentT } from 'whitebox-engine/dists'
import { key } from 'whitebox-engine/random'
import { toArray } from 'whitebox-engine/tensor'
import { boxPlot, cartesian, violinPlot } from 'whitebox-render/charts'
import { figure } from 'whitebox-render/figure'
import { Chart, Choice, ControlPanel, useColourMode } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Box and violin plots',
  question: 'How do I compare the spread of several samples?',
  explain:
    '`boxPlot(groups)` draws quartiles, whiskers at 1.5 IQR and the outliers beyond them; `violinPlot(groups)` mirrors a kernel density estimate of each group. Both take `{ name, values }` groups and are layers, so they can share axes with others.',
}

// The fixed figure the chart fills (the builders' default size).
const FIGURE = figure()

export default function BoxAndViolin(): ReactNode {
  const mode = useColourMode()
  const [kind, setKind] = useState<'box' | 'violin'>('box')
  const groups = useMemo(() => {
    const draw = (d: ReturnType<typeof normal>, i: number): number[] => toArray(d.sample(key(i), [200])) as number[]
    return [
      { name: 'normal', values: draw(normal(0, 1), 1) },
      { name: 'Student t (4)', values: draw(studentT(4), 2) },
      { name: 'gamma (2, 1) − 2', values: draw(gamma(2, 1), 3).map((v) => v - 2) },
    ]
  }, [])
  const option = useMemo(
    () =>
      cartesian([kind === 'box' ? boxPlot(groups, { mode }) : violinPlot(groups, { mode })], {
        mode,
        y: { label: 'value' },
      }),
    [groups, kind, mode],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel>
        <Choice label="plot" value={kind} onChange={setKind} options={['box', 'violin']} />
      </ControlPanel>
      <Chart option={option} figure={FIGURE} />
    </div>
  )
}
