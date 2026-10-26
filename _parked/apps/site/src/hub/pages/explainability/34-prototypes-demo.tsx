import { digits, iris } from 'mdml-engine/datasets'
import { recordsOf } from 'mdml-engine/exposition'
import { prototypesAndCriticisms } from 'mdml-engine/methods/explain'
import { Choice, ControlPanel, NumberField, PrototypeView } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../../gallery/recipe.ts'

export const recipe: Recipe = {
  title: 'Prototypes and criticisms',
  question: 'How do I show the rows that represent a dataset, and those it represents badly?',
  explain:
    '`prototypesAndCriticisms` runs MMD-critic: prototypes are chosen one by one to make the kernel mean of the chosen rows match the data’s, and criticisms are where the witness function (data density minus prototype density) is largest, the rows the prototypes miss. `PrototypeView` marks them on the data’s first two principal components, or, for the digits, draws them as images.',
}

// region
type Data = 'iris' | 'digits'

export default function PrototypeRecipe(): ReactNode {
  const [which, setWhich] = useState<Data>('iris')
  const [count, setCount] = useState(6)
  const record = useMemo(() => {
    const r =
      which === 'iris'
        ? prototypesAndCriticisms(iris(), { nPrototypes: count, nCriticisms: 3 })
        : prototypesAndCriticisms(digits(), { nPrototypes: count, nCriticisms: 3, maxRows: 300, imageShape: [8, 8] })
    return recordsOf(r.exposition, 'prototypes')[0]
  }, [which, count])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Choice label="data" value={which} onChange={setWhich} options={['iris', 'digits'] as Data[]} />
        <NumberField label="prototypes" value={count} onChange={setCount} min={1} max={12} integer />
      </ControlPanel>
      {record !== undefined && <PrototypeView record={record} />}
    </div>
  )
}
// endregion
