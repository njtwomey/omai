import { type ReactNode, useMemo } from 'react'
import { type StructureOverlay, structure } from 'whitebox-engine/scene'
import { Player, StructureView, usePlayback } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Computation graph',
  question: 'How do I animate values flowing through a graph?',
  explain:
    'A structure overlay marks `active` nodes and edges and sets value `badges`; one overlay per frame, played with `usePlayback` and `Player`, animates a forward pass through f(x, y) = (x + y)·sin(y) node by node, in topological order. Passing every `frames` sizes the nodes once, so the layout stays put as values appear.',
}

const g = structure(
  'computation-graph',
  [
    { id: 'x', label: 'x', role: 'input' },
    { id: 'y', label: 'y', role: 'input' },
    { id: 'add', label: '+', role: 'operation' },
    { id: 'sin', label: 'sin', role: 'operation' },
    { id: 'mul', label: '×', role: 'operation' },
    { id: 'f', label: 'f', role: 'output' },
  ],
  [
    { from: 'x', to: 'add' },
    { from: 'y', to: 'add' },
    { from: 'y', to: 'sin' },
    { from: 'add', to: 'mul' },
    { from: 'sin', to: 'mul' },
    { from: 'mul', to: 'f' },
  ],
  { direction: 'right' },
)

// The forward pass at x = 2, y = 0.5, one node per frame.
const values: Record<string, number> = { x: 2, y: 0.5 }
values['add'] = 2.5
values['sin'] = Math.sin(0.5)
values['mul'] = values['add'] * values['sin']
values['f'] = values['mul']
const order = ['x', 'y', 'add', 'sin', 'mul', 'f']

function frame(t: number): StructureOverlay {
  const done = order.slice(0, t + 1)
  const now = order[t] ?? ''
  return {
    active: [now],
    edges: g.edges.filter((e) => e.to === now),
    badges: Object.fromEntries(done.map((id) => [id, { value: Number((values[id] ?? 0).toFixed(3)) }])),
  }
}

const frames = order.map((_, t) => frame(t))

export default function ComputationGraph(): ReactNode {
  const playback = usePlayback(order.length, { fps: 2 })
  const overlay = useMemo(() => frame(playback.frame), [playback.frame])
  return (
    <div className="flex w-fit max-w-full flex-col gap-3">
      <StructureView structure={g} overlay={overlay} frames={frames} frame={{ filename: 'forward-pass', playback }} />
      <Player controller={playback} stamp={(f) => `node ${f + 1} of ${order.length}: ${order[f] ?? ''}`} />
    </div>
  )
}
