import {
  type Paint,
  type Structure,
  type StructureEdge,
  type StructureGroup,
  type StructureNode,
  structure,
} from 'omai-engine/scene'
import { Choice, ControlPanel, StructureView } from 'omai-render/react'
import { type ReactNode, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Looks of a diagram',
  question:
    'How do I draw a diagram that looks designed: pale tinted regions, bold pill titles, typeset labels that fit, and notes tied to their steps?',
  explain:
    'A group’s `look` takes `fillOpacity` (a pale tint under a solid frame), `labelWeight`, `labelSize` and `labelPosition`; a node’s `look` takes the `pill` shape, `fontWeight`, `fontSize` and opacities; an edge’s `look` takes `labelColour`, `labelSize` and `labelWeight`. A `$...$` label is sized by its typeset width, so circles and boxes fit the formula. A `note` may be a record: `{ text, position: "above", tie: "dashed", colour: "fg" }` puts ink above its node with a dashed line to it, and the frame of the group holds it.',
}

const U = 40
const at = (x: number, y: number): readonly [number, number] => [x * U, y * U]
type Tone = 'tab0' | 'tab1' | 'tab2'

const circleOf = (
  id: string,
  label: string,
  x: number,
  y: number,
  tone: Tone,
  group: string | undefined,
  extra: Partial<StructureNode> = {},
  dashed = false,
): StructureNode => ({
  id,
  label,
  role: 'latent',
  ...(group === undefined ? {} : { group }),
  position: at(x, y),
  look: {
    shape: 'circle',
    fill: tone,
    fillOpacity: 0.14,
    stroke: tone,
    text: 'fg',
    strokeWidth: 2.5,
    fontSize: 14,
    ...(dashed ? { dash: [6, 4] } : {}),
  },
  ...extra,
})

const boxOf = (id: string, label: string, x: number, y: number, tone: Paint, group: string): StructureNode => ({
  id,
  label,
  role: 'operation',
  group,
  position: at(x, y),
  look: { shape: 'round', fill: tone, fillOpacity: 0.14, stroke: tone, text: 'fg', strokeWidth: 2.5, fontSize: 14 },
})

/** A bold title in a pill, tinted like its region. */
const title = (id: string, label: string, x: number, y: number, tone: Tone, group: string): StructureNode => ({
  id,
  label,
  role: 'other',
  group,
  position: at(x, y),
  look: {
    shape: 'pill',
    fill: tone,
    fillOpacity: 0.22,
    stroke: tone,
    strokeWidth: 1.5,
    text: 'fg',
    fontWeight: 'bold',
    fontSize: 13,
  },
})

const flow = (from: string, to: string, extra: Partial<StructureEdge> = {}): StructureEdge => ({
  from,
  to,
  ...extra,
  look: { stroke: 'fg', strokeWidth: 2, arrow: true, labelColour: 'fg', labelSize: 12, ...extra.look },
})

const region = (id: string, label: string, tone: Tone, dashed = false): StructureGroup => ({
  id,
  label,
  kind: 'cluster',
  look: {
    stroke: tone,
    fill: tone,
    fillOpacity: 0.07,
    strokeWidth: 2,
    text: tone,
    labelWeight: 'bold',
    labelSize: 13,
    ...(dashed ? { dash: [7, 5] } : { dash: [] }),
  },
})

/** The parts of a diffusion model as one flow: forward chain, network and loss, reverse chain, laid out by hand. */
function diffusion(): Structure {
  const ink = { colour: 'fg' as const, size: 12, tie: 'dashed' as const }
  return structure(
    'flow',
    [
      circleOf('f0', '$\\mathbf{x}_0$', 0, 0, 'tab0', 'forward', { note: { text: 'data', position: 'above', ...ink } }),
      circleOf('ft', '$\\mathbf{x}_t$', 8, 0, 'tab0', 'forward'),
      circleOf(
        'fT',
        '$\\mathbf{x}_T$',
        16,
        0,
        'tab0',
        'forward',
        { note: { text: '$\\approx \\mathcal{N}(\\mathbf{0}, \\mathbf{I})$', position: 'above', ...ink } },
        true,
      ),
      circleOf('eps', '$\\epsilon$', 12.2, 2.2, 'tab1', undefined, {}, true),
      boxOf('net', 'network\n$\\epsilon_\\theta(\\mathbf{x}_t, t)$', 8, 4.2, 'tab1', 'training'),
      boxOf('loss', 'loss\n$\\lVert \\epsilon - \\epsilon_\\theta \\rVert^2$', 14.4, 4.2, 'tab3', 'training'),
      title('t-training', 'TRAINING', 11.2, 6, 'tab1', 'training'),
      circleOf(
        'rT',
        '$\\mathbf{x}_T$',
        16,
        8.6,
        'tab2',
        'reverse',
        { note: { text: '$\\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{I})$', position: 'below', ...ink } },
        true,
      ),
      circleOf('rt', '$\\mathbf{x}_t$', 8, 8.6, 'tab2', 'reverse'),
      circleOf('r0', '$\\mathbf{x}_0$', 0, 8.6, 'tab2', 'reverse', {
        note: { text: 'sample', position: 'below', ...ink },
      }),
    ],
    [
      flow('f0', 'ft', { label: '$q(\\mathbf{x}_t | \\mathbf{x}_0)$\nshrink, add noise', look: { dash: [8, 6] } }),
      flow('ft', 'fT', { label: 'more noise', look: { dash: [8, 6] } }),
      flow('eps', 'ft', { via: [at(8.3, 2.2)] }),
      flow('ft', 'net'),
      flow('eps', 'loss'),
      flow('net', 'loss'),
      flow('net', 'rt'),
      flow('rT', 'rt', { label: 'remove a little noise', look: { dash: [8, 6] } }),
      flow('rt', 'r0', { label: 'repeat down to t = 0', look: { dash: [8, 6] } }),
    ],
    {
      direction: 'right',
      groups: [
        region('forward', 'FORWARD PROCESS: FIXED, NOTHING TO LEARN', 'tab0'),
        region('training', '', 'tab1', true),
        region('reverse', 'REVERSE PROCESS: SAMPLING', 'tab2'),
      ],
      title: 'A diffusion model',
    },
  )
}

export default function DiagramLooks(): ReactNode {
  const [route, setRoute] = useState<'straight' | 'orthogonal'>('orthogonal')
  return (
    <div className="flex w-fit max-w-full flex-col gap-3">
      <ControlPanel>
        <Choice label="route" value={route} options={['straight', 'orthogonal']} onChange={setRoute} />
      </ControlPanel>
      <StructureView structure={diffusion()} layout="manual" route={route} frame={{ filename: 'diagram-looks' }} />
    </div>
  )
}
