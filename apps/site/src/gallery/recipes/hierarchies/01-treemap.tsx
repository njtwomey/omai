import { type TreeNode, treemapChart } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import { Chart, ControlPanel, FigureFrame, NumberField, useColourMode } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Treemap',
  question: 'How do I show a hierarchy of parts by their size?',
  explain:
    '`treemapChart(nodes)` draws every leaf as a rectangle with area proportional to its value, nested in its branch, coloured by top-level branch; click a branch to zoom in and use the trail beneath to come back. Grow a department and watch the others give way.',
}

const FIGURE = figure({ width: 760, height: 460 })
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

export default function Treemap(): ReactNode {
  const mode = useColourMode()
  const [research, setResearch] = useState(1)
  const option = useMemo(() => {
    const nodes: TreeNode[] = BUDGET.map((n) =>
      n.name === 'Research' && n.children !== undefined
        ? { name: n.name, children: n.children.map((c) => ({ name: c.name, value: (c.value ?? 0) * research })) }
        : n,
    )
    return treemapChart(nodes, { mode })
  }, [research, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField
          label="research budget ×"
          value={research}
          onChange={setResearch}
          min={0.25}
          max={4}
          step={0.25}
          log
        />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="treemap">
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
    </div>
  )
}
