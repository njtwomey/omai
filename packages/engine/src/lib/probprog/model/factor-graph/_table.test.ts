// The order of the mapping tables is the meaning of the compiler: a reordering must be a decision, so the order is written here.
import { describe, expect, it } from 'vitest'
import type { FamilyKind } from '../../../message-passing/index.ts'
import { familyHasNoMessageFamily, observedFamilyUnmapped } from './_reasons.ts'
import { latentMappings, observedElementMappings, observedNodeMappings, observedRowMappings } from './_table.ts'

const names = (table: readonly { readonly name: string }[]): string[] => table.map((e) => e.name)

describe('the mapping tables', () => {
  it('asks the latent mappings in this order', () => {
    expect(names(latentMappings)).toEqual([
      'constraint-refused',
      'projected-prior',
      'wishart',
      'correlation',
      'bivariate-normal-refused',
      'mvn-wishart',
      'normal-gamma-mean',
      'normal-gamma-precision',
      'family-refused',
      'normal',
      'multivariate-normal',
      'spike-and-slab',
      'discrete',
      'dirichlet-random-concentration',
      'dirichlet',
      'beta-random-shapes',
      'gamma-random-rate',
      'known-prior',
    ])
  })

  it('asks the whole-node and row mappings of an observed node in this order', () => {
    expect(names(observedNodeMappings)).toEqual([
      'mvn-wishart-observed',
      'bivariate-normal-observed',
      'value-moved-to-mean',
      'rows',
    ])
    expect(names(observedRowMappings)).toEqual(['normal-gamma-observed', 'plate', 'bernoulli-table', 'elements'])
  })

  it('asks the mappings of an element in this order within a family', () => {
    const order = (family: string): string[] =>
      observedElementMappings.filter((e) => e.families?.includes(family)).map((e) => e.name)
    expect(order('normal')).toEqual([
      'normal-switch-log-scale',
      'normal-switched',
      'normal-log-scale',
      'normal-random-scale',
      'normal-random-precision',
      'normal',
    ])
    expect(order('binomial')).toEqual(['binomial-poisson-numeric', 'binomial-poisson-conjugate'])
    expect(order('categorical')).toEqual(['multiclass-logits', 'categorical-conjugate'])
    expect(order('negativeBinomial')).toEqual(['negative-binomial-weibull-numeric', 'negative-binomial-failures'])
    expect(order('weibull')).toEqual(['negative-binomial-weibull-numeric', 'weibull-unmapped'])
    expect(order('orderedProbit')).toEqual(['ordered-numeric', 'ordered-unmapped'])
    expect(order('zeroTruncatedPoisson')).toEqual(['zero-truncated-poisson-numeric', 'zero-truncated-poisson-unmapped'])
    expect(observedElementMappings.at(-1)?.name).toBe('family-unmapped')
  })

  it('names every entry once, says what it maps, and ends each table with an entry that takes every node', () => {
    for (const table of [latentMappings, observedNodeMappings, observedRowMappings, observedElementMappings]) {
      expect(new Set(names(table)).size).toBe(table.length)
      for (const e of table) expect(e.maps.length).toBeGreaterThan(10)
      expect(table.at(-1)?.matches).toBeUndefined()
    }
  })

  it('lists in the reason of an unmapped family exactly the families the tables map', () => {
    // The sentences name what the compiler maps, so a family added to a table without the sentence fails here.
    const sentence = observedFamilyUnmapped('y', 'x').message
    const listed = sentence.slice(sentence.indexOf('mapped are ') + 'mapped are '.length).split(/, | and /)
    const mapped = new Set(observedElementMappings.flatMap((e) => e.families ?? []))
    expect(new Set(listed)).toEqual(mapped)
    const everyFamily: Record<FamilyKind, true> = {
      gaussian: true,
      'gaussian-mv': true,
      gamma: true,
      beta: true,
      bernoulli: true,
      discrete: true,
      dirichlet: true,
      'normal-gamma': true,
      wishart: true,
      'normal-wishart': true,
    }
    const named = familyHasNoMessageFamily('y', 'x').message
    for (const kind of Object.keys(everyFamily)) expect(named).toContain(kind)
  })
})
