// Small datasets for the smoke tests of the supervised methods. Node only; not published.
import type { ClassificationData, Knobs, RegressionData } from '../lib/data/index.ts'
import { generate } from '../lib/data/index.ts'
import { datasetGenerators } from '../lib/datasets/index.ts'
import { key } from '../lib/random/index.ts'

/** A generated classification dataset by generator name (moons, blobs, circles, xor, ...). */
export function classificationSet(name: string, knobs: Knobs = {}, seed = 0): ClassificationData {
  const generator = datasetGenerators[name]
  if (generator === undefined) throw new Error(`no generator ${name}`)
  const { data } = generate(generator, { key: key(seed), knobs })
  if (data.kind !== 'classification') throw new Error(`${name} is not a classification generator`)
  return data
}

/** A generated regression dataset by generator name (sineCurve, friedman1, makeRegression, ...). */
export function regressionSet(name: string, knobs: Knobs = {}, seed = 0): RegressionData {
  const generator = datasetGenerators[name]
  if (generator === undefined) throw new Error(`no generator ${name}`)
  const { data } = generate(generator, { key: key(seed), knobs })
  if (data.kind !== 'regression') throw new Error(`${name} is not a regression generator`)
  return data
}
