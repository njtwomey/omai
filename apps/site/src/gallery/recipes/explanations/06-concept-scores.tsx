import { tcav } from 'mdml-engine/explain'
import { conceptScores } from 'mdml-engine/exposition'
import { matmul, tensor, type Value } from 'mdml-engine/tensor'
import { ControlPanel, ExpositionView, NumberField, SeedInput } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { gaussianRows } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Concept scores',
  question:
    'How do I show whether a concept (stripes, dots, grass) matters to a class, and whether that is more than chance?',
  explain:
    'A `concepts` record holds, per concept, the TCAV score of each run against a random set, the same scores for random-against-random, their mean and a p-value. `ConceptView` draws each concept’s mean score as a bar beside the mean of its random runs, a dot per run, and stars a concept whose p-value is under alpha; a bar far from the random bar and the 0.5 line matters to the class, one near it does not. Here a made-up layer of eight units feeds a linear head; the concepts lie along the head’s direction (`striped`), against it (`grassy`), across it (`dotted`), or nowhere (`blank`). Weaken the concepts or take fewer random sets and the p-values grow.',
  shows: ['concepts', 'ConceptView'],
}

const D = 8
const HEAD = [1, 0.6, 0, -0.5, 0, 0, 0.3, 0]
const norm = Math.hypot(...HEAD)
const tail = (h: Value): Value =>
  matmul(
    h,
    HEAD.map((w) => [w]),
  )
const along = (sign: number): number[] => HEAD.map((w) => (sign * w) / norm)
const DIRECTIONS: readonly (readonly [string, readonly number[]])[] = [
  ['striped', along(1)],
  ['grassy', along(-1)],
  ['dotted', [0, 0, 0, 0, 1, 0, 0, 0]],
  ['blank', new Array<number>(D).fill(0)],
]

export default function ConceptScoresRecipe(): ReactNode {
  const [strength, setStrength] = useState(2)
  const [randomSets, setRandomSets] = useState(20)
  const [seed, setSeed] = useState(1)
  const record = useMemo(() => {
    const cls = tensor(gaussianRows(seed, 40, D))
    const randoms = Array.from({ length: randomSets + 1 }, (_, i) => tensor(gaussianRows(seed * 1000 + 17 + i, 30, D)))
    const concepts = DIRECTIONS.map(([name, dir], c) => {
      const act = gaussianRows(seed * 1000 + 500 + c, 30, D).map((r) => r.map((v, j) => v + strength * (dir[j] ?? 0)))
      const r = tcav(tail, cls, tensor(act), randoms)
      return {
        name,
        scores: [...r.scores],
        randomScores: [...r.randomScores],
        mean: r.mean,
        pValue: r.pValue,
        accuracy: r.accuracies.reduce((a, b) => a + b, 0) / r.accuracies.length,
      }
    })
    return conceptScores({ method: 'TCAV', target: 'zebra', layer: 'layer 4', concepts, alpha: 0.05 })
  }, [strength, randomSets, seed])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="concept strength" value={strength} onChange={setStrength} min={0} max={4} step={0.25} />
        <NumberField label="random sets" value={randomSets} onChange={setRandomSets} min={3} max={60} integer />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <ExpositionView record={record} title="TCAV scores for the class zebra" />
    </div>
  )
}
