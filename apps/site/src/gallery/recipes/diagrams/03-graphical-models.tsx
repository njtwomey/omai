import { structure } from 'mdml-engine/scene'
import { Choice, ControlPanel, StructureView } from 'mdml-render/react'
import { type ReactNode, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Graphical models',
  question: 'How do I draw a graphical model with plates?',
  explain:
    'A `Structure` lists nodes with roles, edges and groups; `StructureView` lays it out and draws it in the language of its `type`: latent variables as open circles, observed ones shaded, factors as small squares, plates around repeated variables with their count (nested by `parent`). Labels written `$...$` are TeX. Drag to pan, Ctrl or Cmd with the wheel to zoom, hover for details.',
}

// region
const mixture = structure(
  'bayes-net',
  [
    { id: 'alpha', label: '$\\alpha$', role: 'parameter', detail: 'Dirichlet concentration' },
    { id: 'pi', label: '$\\pi$', role: 'latent', detail: 'mixing weights' },
    { id: 'z', label: '$z_n$', role: 'latent', group: 'N', detail: 'component of point n' },
    { id: 'x', label: '$x_n$', role: 'observed', group: 'N', detail: 'observed point' },
    { id: 'mu', label: '$\\mu_k$', role: 'latent', group: 'K', detail: 'component mean' },
    { id: 'sigma', label: '$\\Sigma_k$', role: 'latent', group: 'K', detail: 'component covariance' },
  ],
  [
    { from: 'alpha', to: 'pi' },
    { from: 'pi', to: 'z' },
    { from: 'z', to: 'x' },
    { from: 'mu', to: 'x' },
    { from: 'sigma', to: 'x' },
  ],
  {
    groups: [
      { id: 'N', label: 'points', kind: 'plate', count: 'N' },
      { id: 'K', label: 'components', kind: 'plate', count: 'K' },
    ],
  },
)

const factorGraph = structure(
  'factor-graph',
  [
    { id: 'a', label: 'a', role: 'latent' },
    { id: 'b', label: 'b', role: 'latent' },
    { id: 'c', label: 'c', role: 'observed' },
    { id: 'd', label: 'd', role: 'latent' },
    { id: 'f1', label: 'f₁(a, b)', role: 'factor' },
    { id: 'f2', label: 'f₂(b, c)', role: 'factor' },
    { id: 'f3', label: 'f₃(b, d)', role: 'factor' },
  ],
  [
    { from: 'a', to: 'f1', directed: false },
    { from: 'b', to: 'f1', directed: false },
    { from: 'b', to: 'f2', directed: false },
    { from: 'c', to: 'f2', directed: false },
    { from: 'b', to: 'f3', directed: false },
    { from: 'd', to: 'f3', directed: false },
  ],
)

export default function GraphicalModels(): ReactNode {
  const [model, setModel] = useState<'mixture' | 'factor graph'>('mixture')
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel>
        <Choice label="model" value={model} onChange={setModel} options={['mixture', 'factor graph']} />
      </ControlPanel>
      <StructureView structure={model === 'mixture' ? mixture : factorGraph} frame={{ filename: 'graphical-model' }} />
    </div>
  )
}
// endregion
