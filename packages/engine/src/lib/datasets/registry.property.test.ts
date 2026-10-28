import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import type { DatasetKind } from '../data/index.ts'
import { datasetRegistry } from './catalogue.ts'
import {
  DATASET_MODALITIES,
  type DatasetForm,
  type DatasetModality,
  type DatasetQuery,
  matchesDatasetQuery,
  whereDatasets,
} from './registry.ts'

const KINDS: readonly DatasetKind[] = [...new Set(datasetRegistry.entries.map((e) => e.kind))]
const FORMS: readonly DatasetForm[] = ['generator', 'vendored', 'remote']
const WORDS = ['moons', 'real', 'graph', 'two', 'process', 'xyzzy', 'poisson']

const oneOrMany = <T>(values: readonly T[]): fc.Arbitrary<T | readonly T[]> =>
  fc.oneof(
    fc.constantFrom(...values),
    fc.subarray([...values], { minLength: 1 }).map((a) => a as readonly T[]),
  )

const query: fc.Arbitrary<DatasetQuery> = fc.record(
  {
    kind: oneOrMany(KINDS),
    modality: oneOrMany(DATASET_MODALITIES),
    real: fc.boolean(),
    form: oneOrMany(FORMS),
    text: fc.array(fc.constantFrom(...WORDS), { maxLength: 2 }).map((w) => w.join(' ')),
  },
  { requiredKeys: [] },
)

const includes = <T>(filter: T | readonly T[] | undefined, value: T): boolean =>
  filter === undefined || (Array.isArray(filter) ? (filter as readonly T[]).includes(value) : filter === value)

describe('dataset queries', () => {
  it('return exactly the entries whose tags satisfy every filter, in registration order', () => {
    fc.assert(
      fc.property(query, (q) => {
        const found = whereDatasets(datasetRegistry, q)
        const expected = datasetRegistry.entries.filter(
          (e) =>
            includes(q.kind, e.kind) &&
            includes(q.modality, e.modality as DatasetModality) &&
            (q.real === undefined || q.real === e.real) &&
            includes(q.form, e.form) &&
            (q.text ?? '')
              .split(' ')
              .filter((w) => w.length > 0)
              .every((w) =>
                [e.name, e.title, e.description, e.kind, e.modality, e.source ?? '']
                  .join('\n')
                  .toLowerCase()
                  .includes(w),
              ),
        )
        expect(found).toEqual(expected)
        for (const e of found) expect(matchesDatasetQuery(e, q)).toBe(true)
      }),
      { numRuns: 100 },
    )
  })

  it('narrow monotonically: adding a filter never adds entries', () => {
    fc.assert(
      fc.property(query, fc.constantFrom(...KINDS), (q, kind) => {
        const { kind: _dropped, ...rest } = q
        const broad = new Set(whereDatasets(datasetRegistry, rest).map((e) => e.name))
        for (const e of whereDatasets(datasetRegistry, { ...rest, kind })) expect(broad.has(e.name)).toBe(true)
      }),
      { numRuns: 50 },
    )
  })
})
