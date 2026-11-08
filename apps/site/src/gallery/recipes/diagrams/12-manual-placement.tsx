import { type Paint, type Structure, structure } from 'omai-engine/scene'
import { Choice, ControlPanel, StructureView } from 'omai-render/react'
import { type ReactNode, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Placing nodes by hand',
  question:
    'How do I put nodes where I want them: a hub with a ring around it, labels on several lines, my own colours, bent edges and notes?',
  explain:
    'A node with a `position` (its centre) goes there when the layout is `manual`, which is also the default once every node has one. `\\n` in a label or a `detail` starts a new line; a node’s `look` overrides its role’s (the shape `stack` piles boxes behind the first, `none` is a free-standing caption with no outline) and an edge’s `look` its kind’s, and a group’s `look` its frame (colour, fill, dash, label colour). An edge’s `via` lists waypoints it bends at, its label may be TeX (`$q(x_t \\mid x_0)$`) on several lines, a node’s `note` sits under it (or beside it, where an edge would cross), and a factor’s label moves off the edges.',
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

/** A noising chain with a coloured group, TeX labels, a skip edge that bends at waypoints, notes and a factor. */
function noising(): Structure {
  const x = (i: number): readonly [number, number] => [i * 190, 0]
  return structure(
    'bayes-net',
    [
      {
        id: 'x0',
        label: '$x_0$',
        role: 'observed',
        position: x(0),
        group: 'forward',
        detail: 'a sample\nfrom the data',
        look: { shape: 'ellipse' },
        note: '$x_0 \\sim p_{\\text{data}}$',
      },
      { id: 'x1', label: '$x_1$', role: 'latent', position: x(1), group: 'forward', note: 'a little noise' },
      {
        id: 'x2',
        label: '$x_2$',
        role: 'latent',
        position: x(2),
        note: 'almost pure noise\n$\\approx \\mathcal{N}(0, I)$',
      },
      { id: 'f', label: '$\\psi(x_0, x_1)$', role: 'factor', position: [95, -110] },
    ],
    [
      { from: 'x0', to: 'x1', label: '$q(x_1 \\mid x_0)$' },
      { from: 'x1', to: 'x2', label: '$q(x_2 \\mid x_1)$' },
      {
        from: 'x2',
        to: 'x0',
        label: 'reverse\n$p_\\theta(x_0 \\mid x_2)$',
        via: [
          [380, 95],
          [0, 95],
        ],
        look: { stroke: 'tab1', dash: [5, 4] },
      },
      { from: 'x0', to: 'f', look: { arrow: false } },
      { from: 'x1', to: 'f', look: { arrow: false } },
    ],
    {
      groups: [
        {
          id: 'forward',
          label: 'forward process',
          kind: 'cluster',
          look: { stroke: 'tab4', dash: [6, 3], strokeWidth: 1.5, fill: 'surface', text: 'tab4' },
        },
      ],
    },
  )
}

export default function ManualPlacement(): ReactNode {
  const [example, setExample] = useState('hub and ring')
  const [layout, setLayout] = useState<'manual' | 'layered'>('manual')
  const [route, setRoute] = useState<'straight' | 'orthogonal'>('straight')
  const ring = example === 'hub and ring'
  return (
    <div className="flex w-fit max-w-full flex-col gap-3">
      <ControlPanel>
        <Choice label="example" value={example} options={['hub and ring', 'noising chain']} onChange={setExample} />
        <Choice label="layout" value={layout} options={['manual', 'layered']} onChange={setLayout} />
        <Choice label="route" value={route} options={['straight', 'orthogonal']} onChange={setRoute} />
      </ControlPanel>
      <StructureView
        structure={ring ? hubAndRing() : noising()}
        layout={layout}
        route={route}
        frame={{ filename: ring ? 'hub-and-ring' : 'noising-chain' }}
      />
    </div>
  )
}
