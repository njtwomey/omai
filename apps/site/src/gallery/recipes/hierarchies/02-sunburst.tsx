import { sunburstChart, type TreeNode } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import { Chart, ControlPanel, FigureFrame, NumberField, useColourMode } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Sunburst',
  question: 'How do I show a hierarchy as rings?',
  explain:
    '`sunburstChart(nodes)` draws the hierarchy as rings around its root, each node an arc as wide as its share; click a node to put it at the centre. It reads depth better than a treemap and sizes worse, so prefer it when the structure matters more than the amounts.',
}

const FIGURE = figure({ width: 560, height: 560 })
// A toy budget: departments, their teams and spending (in millions).
const BUDGET: TreeNode[] = [
  {
    name: 'Research',
    children: [
      { name: 'Labs', value: 42 },
      { name: 'Grants', value: 25 },
      { name: 'Fellows', value: 9 },
    ],
  },
  {
    name: 'Teaching',
    children: [
      { name: 'Lectures', value: 30 },
      { name: 'Tutors', value: 14 },
      { name: 'Online', value: 7 },
    ],
  },
  {
    name: 'Estates',
    children: [
      { name: 'Buildings', value: 28 },
      { name: 'Energy', value: 12 },
    ],
  },
  {
    name: 'Library',
    children: [
      { name: 'Journals', value: 11 },
      { name: 'Books', value: 4 },
      { name: 'Archives', value: 2 },
    ],
  },
  { name: 'Admin', value: 16 },
]

export default function Sunburst(): ReactNode {
  const mode = useColourMode()
  const [admin, setAdmin] = useState(16)
  const option = useMemo(
    () =>
      sunburstChart(
        BUDGET.map((n) => (n.name === 'Admin' ? { ...n, value: admin } : n)),
        { mode },
      ),
    [admin, mode],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="admin spending" value={admin} onChange={setAdmin} min={0} max={120} integer />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="sunburst">
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
    </div>
  )
}
