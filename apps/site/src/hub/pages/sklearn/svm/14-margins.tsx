import { generate } from 'mdml-engine/data'
import { moons } from 'mdml-engine/datasets'
import { recordsOf } from 'mdml-engine/exposition'
import { svc } from 'mdml-engine/methods/svm'
import { key } from 'mdml-engine/random'
import { Choice, ControlPanel, MarginView, NumberField } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../../../gallery/recipe.ts'

export const recipe: Recipe = {
  title: 'Margins and support vectors',
  question: 'How do I show an SVM’s support vectors, margins and how its solver got there?',
  explain:
    'A support vector classifier returns a `margin` record: its support vectors and dual coefficients, every row’s functional margin y·f(x), SMO’s steps (the pair it updated and the dual objective), the training rows, and the decision surface recorded through training. `MarginView` draws from the record alone and plays the solver: at each recorded step the boundary f = 0, the margins f = ±1 (dashed), the support vectors so far (ringed) and the margins’ histogram, beside the dual objective. A small C tolerates margin violations; a large one chases every row.',
}

// region
const { data } = generate(moons, { knobs: { n: 120, noise: 0.2 }, key: key(6) })

type Kernel = 'rbf' | 'linear' | 'poly'

export default function MarginsRecipe(): ReactNode {
  const [kernel, setKernel] = useState<Kernel>('rbf')
  const [C, setC] = useState(1)
  const fitted = useMemo(() => svc({ kernel, C, gamma: 'scale' }).fit(data), [kernel, C])
  const margin = recordsOf(fitted.exposition, 'margin')[0]
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Choice label="kernel" value={kernel} onChange={setKernel} options={['rbf', 'linear', 'poly'] as Kernel[]} />
        <NumberField label="C" value={C} onChange={setC} min={0.01} max={100} log />
      </ControlPanel>
      {margin !== undefined && <MarginView record={margin} />}
    </div>
  )
}
// endregion
