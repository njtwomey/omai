import { type ReactNode, useMemo, useState } from 'react'
import { decisionTreeClassifier } from 'whitebox-engine/methods/trees'
import type { SpaceValue } from 'whitebox-engine/space'
import { ControlPanel, MethodControls } from 'whitebox-render/react'
import { CodeBlock } from '../../../shell/CodeBlock.tsx'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Controls for a method',
  question: 'How do I give a method’s hyperparameters controls, starting from its demo settings?',
  explain:
    'Estimators declare a `hyperSpace` (for tuning, and for this); `MethodControls` draws it with `SpaceControls`, starting from the hyperparameters the estimator was made with. Feed the result to `estimator.with(hyper)`. The hub builds its pages from exactly this; here the controls only show the estimator they would make.',
}

// region
const base = decisionTreeClassifier({ maxDepth: 3, minSamplesLeaf: 5 })

export default function MethodControlsRecipe(): ReactNode {
  const [hyper, setHyper] = useState<Readonly<Record<string, SpaceValue>> | undefined>(undefined)
  const estimator = useMemo(() => (hyper === undefined ? base : base.with(hyper)), [hyper])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel title={base.name} columns={2}>
        <MethodControls estimator={base} {...(hyper === undefined ? {} : { value: hyper })} onChange={setHyper} />
      </ControlPanel>
      <CodeBlock title="estimator.hyper" code={JSON.stringify(estimator.hyper, null, 2)} />
    </div>
  )
}
// endregion
