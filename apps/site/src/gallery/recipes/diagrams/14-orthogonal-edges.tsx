import { type Structure, structure } from 'omai-engine/scene'
import { Choice, ControlPanel, StructureView } from 'omai-render/react'
import { type ReactNode, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Right-angle edges',
  question: 'How do I draw a flowchart whose edges run along horizontal and vertical lines?',
  explain:
    '`route: "orthogonal"` leaves a node on the side facing its target, runs along horizontal and vertical segments and enters the target square on, with the arrow head along the last segment. The layered layout routes through the channels between its ranks, a long edge down the column of each rank it skips; edges that leave or enter at one port merge, parallel runs get tracks of their own, and a label sits on the middle of a long segment, off the nodes. Straight stays the default, and the right choice for trees, networks and the fan-in of a neural network.',
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

export default function OrthogonalEdges(): ReactNode {
  const [example, setExample] = useState('request handler')
  const [route, setRoute] = useState<'straight' | 'orthogonal'>('orthogonal')
  const [direction, setDirection] = useState<'down' | 'right'>('down')
  return (
    <div className="flex w-fit max-w-full flex-col gap-3">
      <ControlPanel>
        <Choice
          label="example"
          value={example}
          options={['request handler', 'release pipeline']}
          onChange={setExample}
        />
        <Choice label="route" value={route} options={['orthogonal', 'straight']} onChange={setRoute} />
        <Choice label="direction" value={direction} options={['down', 'right']} onChange={setDirection} />
      </ControlPanel>
      <StructureView
        structure={example === 'request handler' ? flowchart() : pipeline()}
        route={route}
        direction={direction}
        frame={{ filename: example === 'request handler' ? 'orthogonal-edges' : 'release-pipeline' }}
      />
    </div>
  )
}
