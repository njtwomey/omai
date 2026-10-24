import { type ReactNode, useMemo, useState } from 'react'
import { generate } from 'whitebox-engine/data'
import { moons } from 'whitebox-engine/datasets'
import type { Exposition } from 'whitebox-engine/exposition'
import { kmeans } from 'whitebox-engine/methods/cluster'
import { adaBoostClassifier } from 'whitebox-engine/methods/ensemble'
import { gaussianNB } from 'whitebox-engine/methods/naive-bayes'
import { svc } from 'whitebox-engine/methods/svm'
import { decisionTreeClassifier } from 'whitebox-engine/methods/trees'
import { key } from 'whitebox-engine/random'
import { toArray } from 'whitebox-engine/tensor'
import { ControlPanel, ExpositionView, Select } from 'whitebox-render/react'
import { viewFor } from 'whitebox-render/views'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Any record',
  question: 'How do I show everything a method explains, without knowing what it returns?',
  explain:
    'Every method returns its `exposition`: a list of records, each with a `kind`. `ExpositionView` draws any record with the standard view of its kind (`viewFor(kind)` names it), given the rows and labels for the views that draw in input space. Pick a method: its records appear one after another, whatever they are.',
}

// region
const { data } = generate(moons, { knobs: { n: 160, noise: 0.25 }, key: key(3) })
const rows = toArray(data.x) as number[][]
const labels = Array.from(data.y.data)

const METHODS = {
  'decision tree': () => decisionTreeClassifier({ maxDepth: 3 }).fit(data).exposition,
  'naive Bayes': () => gaussianNB().fit(data).exposition,
  SVM: () => svc({ C: 1 }).fit(data).exposition,
  AdaBoost: () => adaBoostClassifier({ nEstimators: 20 }).fit(data).exposition,
  'k-means': () => kmeans({ nClusters: 2 }).fit(data, { key: key(0) }).exposition,
} as const satisfies Record<string, () => readonly Exposition[]>

type Method = keyof typeof METHODS

export default function AnyRecordRecipe(): ReactNode {
  const [method, setMethod] = useState<Method>('decision tree')
  // Each record keyed by its kind and how many of that kind came before it.
  const records = useMemo(() => {
    const seen = new Map<string, number>()
    return METHODS[method]().map((record) => {
      const n = seen.get(record.kind) ?? 0
      seen.set(record.kind, n + 1)
      return { id: `${method}-${record.kind}-${n}`, record }
    })
  }, [method])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Select label="method" value={method} onChange={setMethod} options={Object.keys(METHODS) as Method[]} />
      </ControlPanel>
      {records.map(({ id, record: r }) => (
        <section key={id} className="flex flex-col gap-1">
          <h3 className="text-sm font-semibold text-(--wb-fg-secondary)">{viewFor(r.kind).title}</h3>
          <ExpositionView record={r} points={rows} labels={labels} />
        </section>
      ))}
    </div>
  )
}
// endregion
