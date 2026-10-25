import { bool, choice, defaults, dict, int, real, type SpaceValue } from 'mdml-engine/space'
import { ControlPanel, SpaceControls } from 'mdml-render/react'
import { type ReactNode, useState } from 'react'
import { CodeBlock } from '../../../shell/CodeBlock.tsx'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Controls from a space',
  question: 'How do I get controls for a whole set of parameters without writing each one?',
  explain:
    '`SpaceControls` draws one control per dimension of an engine `Space`: reals and integers as number fields (log-scaled as the space says), short choices as segmented controls, long ones as selects, flags as toggles. Conditional dimensions (`when`) appear only while their condition holds: switch the model to see its parameters change.',
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
