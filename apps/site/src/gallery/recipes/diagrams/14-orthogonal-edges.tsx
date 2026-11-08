import { describe as describeNet, mlp } from 'omai-engine/nn'
import { type Structure, structure } from 'omai-engine/scene'
import { Choice, ControlPanel, StructureView } from 'omai-render/react'
import { type ReactNode, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Right-angle edges',
  question: 'How do I draw a flowchart whose edges run along horizontal and vertical lines?',
  explain:
    '`route: "orthogonal"` leaves a node on the side facing its target, runs along horizontal and vertical segments and enters the target square on, with the arrow head along the last segment. The layered layout routes through the channels between its ranks, a long edge down the column of each rank it skips; edges that leave or enter at one port merge when they look the same (colour, width, dash, kind) and get ports side by side when they do not, parallel runs get tracks of their own, and a label sits on the middle of a long segment, off the nodes. An edge may name its `fromSide` and `toSide` (and `fromAt`, `toAt` along them); a node with `obstacle: false`, or an invisible one, is crossed rather than avoided; `cornerRadius` rounds the corners; `labelPlate` sets the plate behind labels (`page`, `none`, or a paint such as the card `surface`); and `origin` pins a position of a manual layout to the same place whichever nodes are shown. Straight stays the default, and the right choice for trees, networks and the fan-in of a neural network.',
}

/** A request handler with a decision, a retry loop, a long edge past three ranks and two edges that merge. */
function flowchart(): Structure {
  const step = (id: string, label: string, detail?: string): Structure['nodes'][number] => ({
    id,
    label,
    role: 'operation',
    look: { shape: 'round', stroke: 'tab0' },
    ...(detail === undefined ? {} : { detail }),
  })
  return structure(
    'pipeline',
    [
      { id: 'start', label: 'request', role: 'input', look: { shape: 'round' } },
      step('parse', 'parse', 'headers and body'),
      { id: 'valid', label: 'valid?', role: 'decision', look: { shape: 'diamond', stroke: 'tab1' } },
      step('reject', 'reject', 'return 400'),
      step('cache', 'cache lookup'),
      step('compute', 'compute'),
      step('log', 'audit log'),
      step('store', 'store result'),
      { id: 'end', label: 'response', role: 'output', look: { shape: 'round' } },
    ],
    [
      { from: 'start', to: 'parse' },
      { from: 'parse', to: 'valid' },
      { from: 'valid', to: 'reject', label: 'no' },
      { from: 'reject', to: 'parse', label: 'retry', look: { stroke: 'tab3', dash: [5, 4] } },
      { from: 'valid', to: 'cache', label: 'yes' },
      { from: 'cache', to: 'compute', label: 'miss' },
      { from: 'cache', to: 'end', label: 'hit', look: { stroke: 'tab2' } },
      { from: 'compute', to: 'store' },
      { from: 'compute', to: 'log' },
      { from: 'store', to: 'end' },
      { from: 'log', to: 'end' },
      { from: 'start', to: 'log', label: 'trace', kind: 'dependency' },
    ],
    { title: 'A request handler' },
  )
}

/** A dense pipeline: fan-out and fan-in, edges that skip stages and one that runs back, so routes could collide. */
function pipeline(): Structure {
  const stage = (id: string, label: string, stroke: string): Structure['nodes'][number] => ({
    id,
    label,
    role: 'block',
    look: { stroke: stroke as 'tab0' },
  })
  const names: [string, string, string][] = [
    ['src', 'source', 'tab0'],
    ['lint', 'lint', 'tab1'],
    ['types', 'typecheck', 'tab1'],
    ['unit', 'unit tests', 'tab1'],
    ['build', 'build', 'tab2'],
    ['linux', 'package linux', 'tab4'],
    ['mac', 'package macOS', 'tab4'],
    ['win', 'package windows', 'tab4'],
    ['sign', 'sign', 'tab3'],
    ['docs', 'docs', 'tab5'],
    ['publish', 'publish', 'tab6'],
    ['notify', 'notify', 'tab6'],
  ]
  const edges = [
    ['src', 'lint'],
    ['src', 'types'],
    ['src', 'unit'],
    ['src', 'build'],
    ['lint', 'build'],
    ['types', 'build'],
    ['build', 'linux'],
    ['build', 'mac'],
    ['build', 'win'],
    ['unit', 'sign'],
    ['linux', 'sign'],
    ['mac', 'sign'],
    ['win', 'sign'],
    ['src', 'docs'],
    ['docs', 'publish'],
    ['sign', 'publish'],
    ['sign', 'notify'],
    ['publish', 'notify'],
    ['win', 'build'],
  ]
  return structure(
    'pipeline',
    names.map(([id, label, stroke]) => stage(id, label, stroke)),
    edges.map(([from, to]) => ({
      from: from as string,
      to: to as string,
      ...(from === 'win' && to === 'build' ? { label: 'rebuild', look: { stroke: 'tab3', dash: [5, 4] } } : {}),
      ...(from === 'unit' ? { label: 'gate' } : {}),
    })),
    { title: 'A release pipeline' },
  )
}

const node = (
  id: string,
  label: string,
  position: readonly [number, number],
  look?: object,
): Structure['nodes'][number] => ({
  id,
  label,
  role: 'block',
  position,
  ...(look === undefined ? {} : { look }),
})

/** Three edges of different colours into one wide box: each gets its own port, nested so none crosses another. */
function wideBox(): Structure {
  return structure(
    'pipeline',
    [
      node('a', 'embed', [0, 0], { stroke: 'tab0' }),
      node('b', 'attend', [0, 70], { stroke: 'tab1' }),
      node('c', 'mix', [0, 140], { stroke: 'tab2' }),
      node('w', 'combine the three streams', [260, 250]),
    ],
    [
      { from: 'a', to: 'w', look: { stroke: 'tab0' } },
      { from: 'b', to: 'w', look: { stroke: 'tab1' } },
      { from: 'c', to: 'w', look: { stroke: 'tab2' } },
    ],
    { title: 'Three edges into one wide box' },
  )
}

/** A pair of different colours into one box, with the sides named: one enters from above, one from below. */
function sides(): Structure {
  return structure(
    'pipeline',
    [
      node('q', 'query', [0, 0], { stroke: 'tab0' }),
      node('k', 'key', [0, 150], { stroke: 'tab1' }),
      node('m', 'match', [220, 75]),
      node('out', 'score', [400, 75]),
    ],
    [
      { from: 'q', to: 'm', toSide: 'top', look: { stroke: 'tab0' }, label: 'query' },
      { from: 'k', to: 'm', toSide: 'bottom', look: { stroke: 'tab1' }, label: 'key' },
      { from: 'm', to: 'out', fromSide: 'right', toSide: 'left' },
    ],
    { title: 'Sides named by the edges' },
  )
}

/** An autoencoder in a row: the dashed edge from the input to the loss runs over the top, on a free side. */
function autoencoderRow(): Structure {
  const names = ['x', 'encoder', 'z', 'decoder', 'x̂', 'loss']
  return structure(
    'pipeline',
    names.map((label, i) => node(label, label, [i * 110, 0], label === 'z' ? { shape: 'round' } : undefined)),
    [
      ...names.slice(1).map((to, i) => ({ from: names[i] as string, to })),
      { from: 'x', to: 'loss', label: 'target', look: { stroke: 'tab3', dash: [5, 4] } },
    ],
    { title: 'An autoencoder, hand placed' },
  )
}

/** The same network from `describe`, with its `target` edge, laid out in ranks. */
const describedAutoencoder = (): Structure => describeNet(mlp([16, 6, 2, 6, 16]), [32, 16], { target: 'reconstruct' })

const EXAMPLES: Readonly<Record<string, () => Structure>> = {
  'request handler': flowchart,
  'release pipeline': pipeline,
  'three into a wide box': wideBox,
  'named sides': sides,
  'autoencoder (placed)': autoencoderRow,
  'autoencoder (describe)': describedAutoencoder,
}

export default function OrthogonalEdges(): ReactNode {
  const [example, setExample] = useState('request handler')
  const [route, setRoute] = useState<'straight' | 'orthogonal'>('orthogonal')
  const [direction, setDirection] = useState<'down' | 'right'>('down')
  const [corners, setCorners] = useState<'sharp' | 'rounded'>('sharp')
  const [plate, setPlate] = useState<'page' | 'surface' | 'none'>('page')
  const [card, setCard] = useState<'page' | 'card'>('page')
  return (
    <div className="flex w-fit max-w-full flex-col gap-3">
      <ControlPanel>
        <Choice label="example" value={example} options={Object.keys(EXAMPLES)} onChange={setExample} />
        <Choice label="route" value={route} options={['orthogonal', 'straight']} onChange={setRoute} />
        <Choice label="direction" value={direction} options={['down', 'right']} onChange={setDirection} />
        <Choice label="corners" value={corners} options={['sharp', 'rounded']} onChange={setCorners} />
        <Choice label="label plate" value={plate} options={['page', 'surface', 'none']} onChange={setPlate} />
        <Choice label="drawn on" value={card} options={['page', 'card']} onChange={setCard} />
      </ControlPanel>
      <div className={card === 'card' ? 'rounded-lg bg-surface p-3' : 'p-3'}>
        <StructureView
          structure={(EXAMPLES[example] ?? flowchart)()}
          route={route}
          direction={direction}
          cornerRadius={corners === 'rounded' ? 8 : 0}
          labelPlate={plate}
          frame={card === 'card' ? false : { filename: example.replaceAll(' ', '-') }}
        />
      </div>
    </div>
  )
}
