import { describe, expect, it } from 'vitest'
import {
  describeGenerator,
  encodeRecipe,
  generate,
  makeFromRecipe,
  parseRecipe,
  rowCount,
  withLabelNoise,
  withMissing,
} from '../data/index.ts'
import { key } from '../random/index.ts'
import { datasetGenerators, makeClassification, moons, remoteDatasets } from './index.ts'

/** Knob values that keep every generator small enough for a quick smoke test. */
const SMALL: Readonly<Record<string, number>> = {
  n: 40,
  users: 40,
  items: 30,
  sessions: 20,
  rounds: 2,
  games: 40,
  nDocs: 20,
  docsPerSlice: 10,
  graphs: 10,
  nodes: 40,
  episodes: 3,
  interactions: 100,
  nOriginals: 10,
  horizon: 20,
}

describe('the dataset catalogue', () => {
  it('labels and describes every knob of every generator', () => {
    for (const g of Object.values(datasetGenerators))
      for (const k of describeGenerator(g).knobs) {
        expect(k.label, `${g.name}.${k.name}`).toBeTypeOf('string')
        expect(k.description?.length ?? 0, `${g.name}.${k.name}`).toBeGreaterThan(5)
      }
  })

  it('generates every dataset, reproducibly, with a recorded recipe', () => {
    const names = Object.keys(datasetGenerators)
    expect(names.length).toBeGreaterThan(70)
    for (const name of names) {
      const g = datasetGenerators[name]
      if (g === undefined) continue
      const knobs: Record<string, number> = {}
      for (const [k, v] of Object.entries(SMALL)) {
        const space = g.knobs.spaces[k]
        if (space !== undefined && (space.kind === 'int' || space.kind === 'real') && v >= space.low && v <= space.high)
          knobs[k] = v
      }
      const a = generate(g, { key: key(1), knobs })
      const b = generate(g, { key: key(1), knobs })
      expect(a.data.kind, name).toBe(g.kind)
      expect(rowCount(a.data), name).toBe(rowCount(b.data))
      expect(JSON.stringify(a.data.meta.recipe?.generator), name).toBe(JSON.stringify(name))
      expect(describeGenerator(g).name).toBe(name)
    }
    expect(Object.keys(remoteDatasets)).toEqual([
      'californiaHousing',
      'electrocardiogram',
      'fashionMnistTest',
      'movielensSmall',
      'tinyShakespeare',
    ])
  })

  it('makes data from recipes with modifiers, through encoding', () => {
    const recipe = parseRecipe(
      encodeRecipe({
        generator: 'makeClassification',
        knobs: { n: 200, nFeatures: 5 },
        seed: 3,
        modifiers: [
          { name: 'labelNoise', knobs: { rate: 0.2 } },
          { name: 'missing', knobs: { mechanism: 'MAR', rate: 0.1 } },
        ],
      }),
    )
    const g = makeFromRecipe(recipe, { generators: datasetGenerators })
    expect(g.data.meta.recipe?.modifiers?.map((m) => m.name)).toEqual(['labelNoise', 'missing'])
    expect(g.truth.latents?.['cleanLabels']).toBeDefined()
    expect(g.truth.latents?.['missing']).toBeDefined()
    // Modifiers compose with generators directly too, keeping the truth.
    const direct = withMissing(withLabelNoise(generate(moons, { key: key(0) }), { rate: 0.1, key: key(1) }), {
      key: key(2),
    })
    expect(direct.truth.posterior).toBeDefined()
    expect(generate(makeClassification, { key: key(4), knobs: { n: 50 } }).data.x.shape).toEqual([50, 20])
  })
})
