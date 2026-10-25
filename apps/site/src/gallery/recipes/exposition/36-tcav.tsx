import { type ReactNode, useMemo, useState } from 'react'
import { digits } from 'whitebox-engine/datasets'
import { recordsOf } from 'whitebox-engine/exposition'
import { takeData } from 'whitebox-engine/learn'
import { tcavStudy } from 'whitebox-engine/methods/explain'
import { lenet } from 'whitebox-engine/methods/neural/vision'
import { key } from 'whitebox-engine/random'
import { asTensor, type TensorLike, toFlat } from 'whitebox-engine/tensor'
import { ConceptView, ControlPanel, NumberField } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Concept scores (TCAV)',
  question: 'How do I test whether a network’s decision for a class rests on a human concept?',
  explain:
    '`tcavStudy` finds, at the LeNet’s hidden layer, the direction separating images of a concept from random images (a concept activation vector), and scores the class by the share of its images whose logit rises along it; random-versus-random directions give the null, and a t test stars the concepts that differ from it. The concepts here are other digits, so a 9’s reliance on 4-like and 7-like strokes can be tested. `ConceptView` draws the mean score per concept against random, each run a dot.',
}

// region
const data = takeData(
  digits(),
  Array.from({ length: 900 }, (_, i) => i),
)
const net = lenet({ epochs: 4, snapshots: 2 }).fit(data, { key: key(1), explain: false })
const labels = toFlat(asTensor(data.y as TensorLike))
const rowsOf = (d: number): number[] => labels.flatMap((v, i) => (v === d ? [i] : [])).slice(0, 40)

export default function TcavRecipe(): ReactNode {
  const [digit, setDigit] = useState(9)
  const record = useMemo(() => {
    const concepts = [0, 1, 4, 7].filter((c) => c !== digit).map((c) => ({ name: `${c}-like`, rows: rowsOf(c) }))
    return recordsOf(
      tcavStudy(net, data, concepts, { classIndex: digit, nRandom: 8, key: key(2) }).exposition,
      'concepts',
    )[0]
  }, [digit])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={1}>
        <NumberField label="class explained" value={digit} onChange={setDigit} min={0} max={9} integer />
      </ControlPanel>
      {record !== undefined && <ConceptView record={record} />}
    </div>
  )
}
// endregion
