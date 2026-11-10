// The chains of a sampler's posterior regrouped by chain and coordinate: the rows are the chains' own draws, a vector site
// splits into `site[i]`, and the divergent flags and energy come with them.
import { describe, expect, it } from 'vitest'
import { normal } from '../dists/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { mcmc } from './engines/index.ts'
import { definePosterior } from './posterior.ts'
import { posteriorChains } from './posterior-chains.ts'
import { observe, sample } from './primitives.ts'

const model = (): void => {
  const mu = sample('mu', normal(0, 1))
  observe('y', normal(mu, 1), 1)
}
const posterior = mcmc(model, undefined, { key: key(3), numChains: 3, numWarmup: 40, numDraws: 30 })

describe('posteriorChains', () => {
  const c = posteriorChains(posterior)
  it('has a row of draws per chain for every scalar coordinate', () => {
    expect(Object.keys(c.draws)).toContain('mu')
    for (const rows of Object.values(c.draws)) {
      expect(rows).toHaveLength(3)
      for (const r of rows) expect(r).toHaveLength(30)
    }
  })

  it('lays the rows out as the posterior keeps them: chain after chain', () => {
    const flat = toFlat(posterior.draws?.['mu'] as never)
    expect(c.draws['mu']?.[1]?.[4]).toBe(flat[30 + 4])
    expect(c.draws['mu']?.[2]?.[29]).toBe(flat[89])
  })

  it('brings the divergent flags and the energy by chain, and only the sites asked for', () => {
    expect(c.divergent).toHaveLength(3)
    expect(c.energy?.[0]).toHaveLength(30)
    expect(Object.keys(posteriorChains(posterior, { sites: ['mu'] }).draws)).toEqual(['mu'])
  })

  it('is empty for a posterior that has no chains', () => {
    expect(posteriorChains(definePosterior({ form: 'exact', engine: 'x', sites: [], marginals: {} })).draws).toEqual({})
  })
})
