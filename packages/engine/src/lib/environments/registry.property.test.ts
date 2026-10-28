import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import {
  type ActionKind,
  ENVIRONMENT_FAMILIES,
  type EnvironmentQuery,
  environmentRegistry,
  matchesEnvironmentQuery,
  type ObservationKind,
  whereEnvironments,
} from './registry.ts'

const ACTIONS: readonly ActionKind[] = ['discrete', 'continuous', 'multi']
const OBSERVATIONS: readonly ObservationKind[] = ['vector', 'grid', 'image', 'discrete']

const oneOrMany = <T>(values: readonly T[]): fc.Arbitrary<T | readonly T[]> =>
  fc.oneof(
    fc.constantFrom(...values),
    fc.subarray([...values], { minLength: 1 }).map((a) => a as readonly T[]),
  )

const query: fc.Arbitrary<EnvironmentQuery> = fc.record(
  {
    family: oneOrMany(ENVIRONMENT_FAMILIES),
    actions: oneOrMany(ACTIONS),
    observations: oneOrMany(OBSERVATIONS),
    episodic: fc.boolean(),
    multiAgent: fc.boolean(),
    text: fc.constantFrom('', 'pole', 'maze', 'v1', 'poker', 'xyzzy'),
  },
  { requiredKeys: [] },
)

describe('environment queries', () => {
  it('return a subset of the registry, in order, every member matching, and nothing outside it matching', () => {
    fc.assert(
      fc.property(query, (q) => {
        const found = whereEnvironments(environmentRegistry, q)
        const names = found.map((e) => e.name)
        const inOrder = environmentRegistry.entries.filter((e) => names.includes(e.name)).map((e) => e.name)
        expect(names).toEqual(inOrder)
        for (const e of environmentRegistry.entries) expect(matchesEnvironmentQuery(e, q)).toBe(names.includes(e.name))
      }),
      { numRuns: 100 },
    )
  })
})
