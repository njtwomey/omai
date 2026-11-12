// The registry as a whole: every operator is keyed by its own name, declares what it supports consistently with the hooks
// it has, and the gate that runs the others is the last. The net under the moves and merges of the operator files.
import { describe, expect, it } from 'vitest'
import { covarianceOperatorList } from './covariance/list.ts'
import { discreteOperatorList } from './discrete/list.ts'
import { gaussianOperatorList } from './gaussian/list.ts'
import { liftOperatorList } from './lifts/list.ts'
import { likelihoodOperatorList } from './likelihoods/list.ts'
import { linkOperatorList } from './links/list.ts'
import { operatorNames } from './names.ts'
import { collectOperators, defaultOperators } from './registry.ts'
import { scaleOperatorList } from './scales/list.ts'

describe('defaultOperators', () => {
  const ops = defaultOperators()
  const names = Object.keys(ops)

  it('registers the 69 operators, the gate last', () => {
    expect(names).toHaveLength(69)
    expect(names.at(-1)).toBe('gate')
    expect(new Set(names).size).toBe(names.length)
    expect(names.slice(0, 3)).toEqual(['prior', 'gaussian', 'gaussian-gamma-precision'])
  })

  it('lists the names as a typed tuple, in registry order', () => {
    expect([...operatorNames]).toEqual(names)
  })

  it('keys every operator by its own name and gives it ports and a description', () => {
    for (const [key, op] of Object.entries(ops)) {
      expect(op.name).toBe(key)
      expect(op.description.length).toBeGreaterThan(0)
      // The ports of nine operators are a function of the parameters (a vector of known length): a static list is not empty.
      if (typeof op.ports !== 'function') expect(op.ports.length, key).toBeGreaterThan(0)
    }
  })

  it('declares support exactly where there is a hook', () => {
    for (const [key, op] of Object.entries(ops)) {
      expect(op.support.ep !== undefined, `${key} ep`).toBe(op.ep !== undefined || op.tilted !== undefined)
      expect(op.support.vmp !== undefined, `${key} vmp`).toBe(op.vmp !== undefined)
    }
  })

  it('gives every operator with an EP rule an evidence rule', () => {
    const without = names.filter((k) => ops[k]?.support.ep !== undefined && ops[k]?.logEvidence === undefined)
    expect(without).toEqual([])
  })

  it('returns a new object each call, so a caller may add its own', () => {
    expect(defaultOperators()).not.toBe(ops)
    expect(Object.keys({ ...defaultOperators(), mine: ops['prior'] })).toContain('mine')
  })

  it('refuses a registry that disagrees with its own list', () => {
    const prior = ops['prior'] as NonNullable<(typeof ops)[string]>
    const equal = ops['equal'] as NonNullable<(typeof ops)[string]>
    expect(Object.keys(collectOperators([equal, prior], ['prior', 'equal']))).toEqual(['prior', 'equal'])
    expect(() => collectOperators([prior, prior], ['prior'])).toThrow("two operators are named 'prior'")
    expect(() => collectOperators([prior], ['prior', 'equal'])).toThrow("'equal' and no operator has it")
    expect(() => collectOperators([prior, equal], ['prior'])).toThrow("'equal' is not in the list of names")
  })

  it('builds from the folders: every operator of every folder list is in the registry', () => {
    const listed = [
      ...gaussianOperatorList,
      ...linkOperatorList,
      ...discreteOperatorList,
      ...liftOperatorList,
      ...likelihoodOperatorList,
      ...scaleOperatorList,
      ...covarianceOperatorList,
    ]
    expect(listed).toHaveLength(operatorNames.length - 1)
    for (const op of listed) expect(ops[op.name], op.name).toBeDefined()
  })
})
