import { type Paint, type Structure, structure } from 'omai-engine/scene'
import { Choice, ControlPanel, StructureView } from 'omai-render/react'
import { type ReactNode, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Placing nodes by hand',
  question:
    'How do I put nodes where I want them: a hub with a ring around it, labels on several lines, my own colours?',
  explain:
    'A node with a `position` (its centre) goes there when the layout is `manual`, which is also the default once every node has one. `\\n` in a label starts a new line; a node’s `look` overrides its role’s (the shape `stack` piles boxes behind the first, `none` is a caption with no outline) and an edge’s `look` its kind’s.',
}

const RING = 6
const tab = (i: number): Paint => `tab${(i + 2) % 10}` as Paint
const R = 120

/** A hub at the centre, a ring of workers around it, a stack of replicas to one side and a caption below. */
function hubAndRing(): Structure {
  const workers = Array.from({ length: RING }, (_, i) => {
    const a = (2 * Math.PI * i) / RING - Math.PI / 2
    return {
      id: `w${i}`,
      label: `worker ${i + 1}\nshard ${i}`,
      role: 'operation' as const,
      position: [R * Math.cos(a), R * Math.sin(a)] as const,
      look: { stroke: tab(i) },
    }
  })
  return structure(
    'architecture',
    [
      {
        id: 'hub',
        label: 'parameter\nserver',
        role: 'block',
        position: [0, 0],
        look: { shape: 'circle', stroke: 'accent', strokeWidth: 2 },
      },
      ...workers,
      { id: 'replicas', label: 'replicas', role: 'tensor', position: [R * 2.3, 0], look: { shape: 'stack' } },
      {
        id: 'note',
        label: 'a hub with a ring of workers',
        role: 'other',
        position: [0, R * 1.5],
        look: { shape: 'none' },
      },
    ],
    [
      ...workers.map((w, i) => ({
        from: 'hub',
        to: w.id,
        bidirectional: true,
        look: { stroke: tab(i) },
      })),
      { from: 'hub', to: 'replicas', label: 'checkpoint', labelAt: 0.5, kind: 'dependency' as const },
    ],
    { title: 'Manual placement' },
  )
}

export default function ManualPlacement(): ReactNode {
  const [layout, setLayout] = useState<'manual' | 'layered'>('manual')
  return (
    <div className="flex w-fit max-w-full flex-col gap-3">
      <ControlPanel>
        <Choice label="layout" value={layout} options={['manual', 'layered']} onChange={setLayout} />
      </ControlPanel>
      <StructureView structure={hubAndRing()} layout={layout} frame={{ filename: 'hub-and-ring' }} />
    </div>
  )
}
