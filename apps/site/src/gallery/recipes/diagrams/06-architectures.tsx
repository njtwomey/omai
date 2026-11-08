import {
  Conv2d,
  describe as describeNet,
  Linear,
  MaxPool2d,
  mlp,
  ReLU,
  Residual,
  Sequential,
  TransformerEncoderLayer,
} from 'omai-engine/nn'
import type { Structure } from 'omai-engine/scene'
import { Choice, ControlPanel, SceneFrame, Select } from 'omai-render/react'
import { type ArchitectureView, architecture } from 'omai-render/structures'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Network architectures',
  question: 'How do I draw a neural network: as a diagram, a stack, its tensor shapes, or where its parameters are?',
  explain:
    '`describe(module, inputShape)` in omai-engine turns a network into a `Structure` (layers with shapes, parameter counts and FLOPs; residual and parallel blocks as groups; `target` draws the input as the output’s target, as in an autoencoder). `architecture(structure, view)` draws it four ways: the full `diagram`, a `compact` stack, the `shapes` of the tensors flowing through, and a `params` treemap.',
}

function residualMlp(): Structure {
  const block = (): Residual => new Residual(new Sequential(new Linear(32, 32), new ReLU(), new Linear(32, 32)))
  return describeNet(new Sequential(new Linear(8, 32), block(), block(), new Linear(32, 2)), [16, 8])
}

/** An autoencoder: the output is compared with the input, drawn as the `target` edge. */
const autoencoder = (): Structure => describeNet(mlp([16, 6, 2, 6, 16]), [32, 16], { target: 'reconstruct' })

const NETWORKS: Readonly<Record<string, () => Structure>> = {
  'multilayer perceptron': () => describeNet(mlp([4, 64, 64, 3]), [32, 4]),
  'residual MLP': residualMlp,
  autoencoder,
  'small CNN': () =>
    describeNet(
      new Sequential(
        new Conv2d(1, 8, 3),
        new ReLU(),
        new MaxPool2d(2),
        new Conv2d(8, 16, 3),
        new ReLU(),
        new MaxPool2d(2),
      ),
      [4, 1, 28, 28],
    ),
  'transformer layer': () =>
    describeNet(new TransformerEncoderLayer(32, 4, { batchFirst: true }), [2, 10, 32], { expandParts: true }),
}

export default function Architectures(): ReactNode {
  const [name, setName] = useState('residual MLP')
  const [view, setView] = useState<ArchitectureView>('diagram')
  const drawing = useMemo(() => architecture((NETWORKS[name] ?? residualMlp)(), view), [name, view])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Select label="network" value={name} onChange={setName} options={Object.keys(NETWORKS)} />
        <Choice label="view" value={view} onChange={setView} options={['diagram', 'compact', 'shapes', 'params']} />
      </ControlPanel>
      <SceneFrame scene={drawing} filename="architecture" maxHeight={960} />
    </div>
  )
}
