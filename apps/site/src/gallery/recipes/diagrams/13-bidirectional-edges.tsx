import { type Structure, structure } from 'omai-engine/scene'
import { Choice, ControlPanel, StructureView } from 'omai-render/react'
import { type ReactNode, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Two-way edges and edge labels',
  question: 'How do I draw an edge that runs both ways, and keep its label off the nodes?',
  explain:
    '`bidirectional: true` draws one line with an arrow head at each end, and an edge written the other way round between the same nodes is absorbed into it, so ranks stay as they would be without the twin. A label sits at the middle of its edge on a plate above the nodes, is moved aside when it would cover a node, or goes where `labelAt` (0 to 1 along the edge) says.',
}

function protocol(twoWay: boolean): Structure {
  const nodes = ['client', 'proxy', 'server', 'cache'].map((id) => ({ id, label: id, role: 'block' as const }))
  const pair = twoWay
    ? [{ from: 'client', to: 'proxy', bidirectional: true, label: 'request / reply' }]
    : [
        { from: 'client', to: 'proxy', label: 'request' },
        { from: 'proxy', to: 'client', label: 'reply' },
      ]
  return structure(
    'architecture',
    nodes,
    [
      ...pair,
      { from: 'proxy', to: 'server', bidirectional: twoWay, label: 'forward' },
      { from: 'proxy', to: 'cache', label: 'lookup', labelAt: 0.3, look: { stroke: 'tab1' } },
      { from: 'cache', to: 'server', label: 'a long label that would sit on the nodes', kind: 'dependency' },
      ...(twoWay ? [] : [{ from: 'server', to: 'proxy', label: 'answer' }]),
    ],
    { direction: 'right' },
  )
}

export default function BidirectionalEdges(): ReactNode {
  const [style, setStyle] = useState('one edge, two heads')
  return (
    <div className="flex w-fit max-w-full flex-col gap-3">
      <ControlPanel>
        <Choice
          label="pairs drawn as"
          value={style}
          options={['one edge, two heads', 'two edges']}
          onChange={setStyle}
        />
      </ControlPanel>
      <StructureView structure={protocol(style === 'one edge, two heads')} frame={{ filename: 'two-way' }} />
    </div>
  )
}
