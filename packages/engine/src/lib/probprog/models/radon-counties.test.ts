import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import { analysisOf } from '../inference/model-input.ts'
import { nutsMethod } from '../inference/nuts-method.ts'
import { generateRadonCounties, radonCountiesDefinition } from './radon-counties.ts'

const data = generateRadonCounties(key(2), { counties: 8, minSize: 1, tau: 0.5, slopeSd: 0.3 })

describe('generateRadonCounties', () => {
  it('has counties of unequal size, none under the smallest, and carries its truth', () => {
    const sizes = data.generating['sizes'] as number[]
    expect(sizes).toHaveLength(8)
    expect(Math.min(...sizes)).toBeGreaterThanOrEqual(1)
    expect(Math.max(...sizes)).toBeGreaterThan(Math.min(...sizes))
    const county = data.data['county'] as number[]
    expect(county).toHaveLength(sizes.reduce((a, b) => a + b, 0))
    expect(new Set(county).size).toBe(8)
    expect(data.generating['mu']).toBe(1.5)
  })

  it('is the same for the same key', () => {
    expect(generateRadonCounties(key(2), { counties: 8, minSize: 1, tau: 0.5, slopeSd: 0.3 })).toEqual(data)
  })
})

describe('radonCountiesDefinition', () => {
  for (const pooling of ['pooled', 'unpooled', 'partial', 'slopes'] as const) {
    it(`${pooling}: binds to flat data of unequal groups and NUTS runs on it`, () => {
      const bound = radonCountiesDefinition(pooling).bind(data.data)
      expect(bound.sizes['county']).toBe(8)
      expect(bound.sizes['house']).toBe((data.data['y'] as number[]).length)
      expect(nutsMethod(bound).check?.(analysisOf(bound)).applies).toBe(true)
    })
  }

  it('the centred and non-centred partial pooling have the county intercepts as a latent and a derived site', () => {
    const kinds = (p: 'centred' | 'non-centred'): string[] =>
      radonCountiesDefinition('partial', p)
        .spec.nodes.filter((n) => n.name === 'alpha')
        .map((n) => n.kind)
    expect(kinds('centred')).toEqual(['latent'])
    expect(kinds('non-centred')).toEqual(['derived'])
  })
})
