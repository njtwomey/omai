import { factor, factorGraph, toStructure } from 'mdml-engine/pgm'
import { Choice, ControlPanel, StructureView } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Factor graphs',
  question: 'How do I draw a factor graph, and show what a variable is connected to?',
  explain:
    'A factor graph from mdml-engine’s `pgm` becomes a structure with `toStructure`: variables as circles, factors as small squares. Factor graphs are drawn `bipartite` by default, variables in one row and factors in the other, so every edge crosses between them; `force` lays them out freely. Click a node to light up its neighbourhood (an overlay of active nodes and edges).',
}

const binary = ['0', '1']
const agree = [2, 1, 1, 2]
const fg = factorGraph(
  ['a', 'b', 'c', 'd', 'e'].map((name) => ({ name, states: binary })),
  [
    factor(['a'], [2], [3, 1]),
    factor(['a', 'b'], [2, 2], agree),
    factor(['b', 'c'], [2, 2], agree),
    factor(['b', 'd'], [2, 2], agree),
    factor(['c', 'd', 'e'], [2, 2, 2], [4, 1, 1, 1, 1, 1, 1, 4]),
  ],
)
const s = toStructure(fg, ['e'])

export default function FactorGraphs(): ReactNode {
  const [layout, setLayout] = useState<'bipartite' | 'force'>('bipartite')
  const [picked, setPicked] = useState('b')
  const overlay = useMemo(() => {
    const edges = s.edges.filter((e) => e.from === picked || e.to === picked)
    const near = edges.map((e) => (e.from === picked ? e.to : e.from))
    return { active: [picked, ...near], edges }
  }, [picked])
  return (
    <div className="flex w-fit max-w-full flex-col gap-3">
      <ControlPanel>
        <Choice label="layout" value={layout} options={['bipartite', 'force'] as const} onChange={setLayout} />
      </ControlPanel>
      <StructureView
        structure={s}
        layout={layout}
        overlay={overlay}
        seed={3}
        width={520}
        onNodeClick={(n) => setPicked(n.id)}
      />
    </div>
  )
}
