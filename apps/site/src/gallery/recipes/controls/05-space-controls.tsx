import { bool, choice, defaults, dict, int, real, type SpaceValue, union } from 'mdml-engine/space'
import { ControlPanel, SpaceControls } from 'mdml-render/react'
import { type ReactNode, useState } from 'react'
import { CodeBlock } from '../../../shell/CodeBlock.tsx'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Controls from a space',
  question: 'How do I get controls for a whole set of parameters without writing each one?',
  explain:
    '`SpaceControls` draws one control per dimension of an engine `Space`: reals and integers as number fields (log-scaled as the space says), short choices as segmented controls, long ones as selects, flags as toggles. Conditional dimensions (`when`) appear only while their condition holds: switch the model to see its parameters change. A `union` is a mode with the knobs each option brings, nested as deep as it goes: switch the solver, then the optimiser inside it.',
}

// region
const space = dict({
  model: choice(['ridge', 'tree', 'knn'], { label: 'model' }),
  alpha: real(1e-4, 10, { log: true, label: 'penalty α', when: { key: 'model', in: ['ridge'] } }),
  depth: int(1, 12, { label: 'max depth', when: { key: 'model', in: ['tree'] } }),
  criterion: choice(['gini', 'entropy', 'log loss'], { when: { key: 'model', in: ['tree'] } }),
  k: int(1, 50, { label: 'neighbours', when: { key: 'model', in: ['knn'] } }),
  metric: choice(['euclidean', 'manhattan', 'chebyshev', 'cosine', 'minkowski'], {
    when: { key: 'model', in: ['knn'] },
  }),
  standardise: bool({ default: true, label: 'standardise inputs' }),
  solver: union(
    {
      closedForm: dict({}),
      irls: dict({
        tol: real(1e-10, 1e-2, { log: true, label: 'tolerance' }),
        maxIter: int(1, 200, { label: 'iterations' }),
      }),
      sgd: dict({
        batch: int(1, 256, { log: true, label: 'batch size' }),
        optimiser: union(
          {
            plain: dict({}),
            momentum: dict({ beta: real(0, 0.99, { label: 'momentum β' }) }),
            adam: dict({ lr: real(1e-4, 1, { log: true }) }),
          },
          { tag: 'name', label: 'optimiser' },
        ),
      }),
    },
    { default: 'irls', label: 'solver' },
  ),
})

export default function SpaceControlsRecipe(): ReactNode {
  const [value, setValue] = useState<SpaceValue>(() => defaults(space))
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <SpaceControls space={space} value={value} onChange={setValue} />
      </ControlPanel>
      <CodeBlock title="value" code={JSON.stringify(value, null, 2)} />
    </div>
  )
}
// endregion
