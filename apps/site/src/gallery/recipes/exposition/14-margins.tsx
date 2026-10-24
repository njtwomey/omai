import { type ReactNode, useMemo, useState } from 'react'
import { generate } from 'whitebox-engine/data'
import { moons } from 'whitebox-engine/datasets'
import { decisionSurface, recordsOf } from 'whitebox-engine/exposition'
import { svc } from 'whitebox-engine/methods/svm'
import { key } from 'whitebox-engine/random'
import { toArray, toFlat } from 'whitebox-engine/tensor'
import { Choice, ControlPanel, MarginView, NumberField } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Margins and support vectors',
  question: 'How do I show an SVM’s support vectors, margins and how its solver got there?',
  explain:
    'A support vector classifier returns a `margin` record: its support vectors and dual coefficients, every row’s functional margin y·f(x), and SMO’s steps (the pair it updated and the dual objective). `MarginView` rings the support vectors over the decision surface (bigger for larger coefficients) and plays the solver. The margins f(x) = ±1 are dashed; a small C tolerates margin violations; a large one chases every row.',
}

// region
const { data } = generate(moons, { knobs: { n: 120, noise: 0.2 }, key: key(6) })
const rows = toArray(data.x) as number[][]
const labels = Array.from(data.y.data)

type Kernel = 'rbf' | 'linear' | 'poly'

export default function MarginsRecipe(): ReactNode {
  const [kernel, setKernel] = useState<Kernel>('rbf')
  const [C, setC] = useState(1)
  const fitted = useMemo(() => svc({ kernel, C, gamma: 'scale' }).fit(data), [kernel, C])
  const margin = recordsOf(fitted.exposition, 'margin')[0]
  // The recorded surface holds labels; f(x) on the same grid gives the smooth boundary and the margins f = ±1.
  const surface = useMemo(() => {
    const s = recordsOf(fitted.exposition, 'decisionSurface')[0]
    if (s === undefined) return undefined
    const grid = s.y.flatMap((b) => s.x.map((a) => [a, b]))
    const f = Array.from(toFlat(fitted.decisionFunction(grid)))
    const values = s.y.map((_, i) => f.slice(i * s.x.length, (i + 1) * s.x.length))
    return decisionSurface({
      features: s.features,
      x: s.x,
      y: s.y,
      values,
      ...(s.classes === undefined ? {} : { classes: s.classes }),
    })
  }, [fitted])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Choice label="kernel" value={kernel} onChange={setKernel} options={['rbf', 'linear', 'poly'] as Kernel[]} />
        <NumberField label="C" value={C} onChange={setC} min={0.01} max={100} log />
      </ControlPanel>
      {margin !== undefined && (
        <MarginView record={margin} points={rows} labels={labels} {...(surface === undefined ? {} : { surface })} />
      )}
    </div>
  )
}
// endregion
