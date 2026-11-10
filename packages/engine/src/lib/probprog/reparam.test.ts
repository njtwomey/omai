import { describe, expect, it } from 'vitest'
import { expectDistribution, normalCdf } from '../../testing/sampling.ts'
import { transformed } from '../dists/compose/index.ts'
import { bijectors, halfCauchy, normal, vonMises } from '../dists/index.ts'
import { foldIn, key as makeKey } from '../random/index.ts'
import { exp, item, tensor, toFlat } from '../tensor/index.ts'
import { logJoint } from './density.ts'
import { mcmc } from './engines/index.ts'
import { seed, traceOf } from './handlers.ts'
import { type Model, observe, plate, sample } from './primitives.ts'
import { circular, locScale, reparam, transformReparam } from './reparam.ts'

const THETA: Model = () => {
  const theta = sample('theta', normal(1.5, 2.5))
  observe('y', normal(theta, 1), 0.7)
}

describe('locScale', () => {
  it('turns the site into an auxiliary draw and a deterministic site', () => {
    const t = traceOf(seed(reparam(THETA, { theta: locScale(0) }), makeKey(0)), undefined)
    expect([...t.sites.keys()]).toEqual(['theta_decentered', 'theta', 'y'])
    expect(t.sites.get('theta')?.type).toBe('deterministic')
    const z = t.sites.get('theta_decentered')
    const theta = t.sites.get('theta')
    if (z?.type !== 'sample' || theta?.type !== 'deterministic') throw new Error('unexpected sites')
    expect(item(theta.value)).toBeCloseTo(1.5 + 2.5 * item(z.value), 14)
  })

  it('keeps the density of the original program up to the Jacobian, for every centring', () => {
    for (const c of [0, 0.3, 0.8]) {
      const re = reparam(THETA, { theta: locScale(c) })
      for (const x of [-2, 0.4, 3.1]) {
        // x = loc + scale^(1 - c) (z - c loc), so z = c loc + (x - loc) / scale^(1 - c).
        const z = c * 1.5 + (x - 1.5) / 2.5 ** (1 - c)
        const original = item(logJoint(THETA, undefined, { theta: x }).logJoint)
        const decentred = item(logJoint(re, undefined, { theta_decentered: z }).logJoint)
        expect(decentred).toBeCloseTo(original + (1 - c) * Math.log(2.5), 12)
      }
    }
  })

  it('draws the original marginal law (Kolmogorov-Smirnov, fixed seeds)', () => {
    const re = reparam(() => sample('theta', normal(1.5, 2.5)), { theta: locScale(0.4) })
    const draws = Array.from({ length: 1500 }, (_, i) => {
      const t = traceOf(seed(re, foldIn(makeKey(5), i)), undefined)
      const s = t.sites.get('theta')
      return s?.type === 'deterministic' ? item(s.value) : Number.NaN
    })
    expectDistribution(draws, (x) => normalCdf((x - 1.5) / 2.5))
  })

  it('broadcasts under plates and per-element scales', () => {
    const model: Model = () => {
      const s = sample('s', normal(0, 1))
      plate('n', 3, () => sample('x', normal(tensor([0, 1, 2]), exp(s))))
    }
    const re = reparam(model, { x: locScale(0) })
    const t = traceOf(seed(re, makeKey(1)), undefined)
    const aux = t.sites.get('x_decentered')
    expect(aux?.type === 'sample' && aux.value.shape).toEqual([3])
    const x = t.sites.get('x')
    const s = t.sites.get('s')
    const z = t.sites.get('x_decentered')
    if (x?.type !== 'deterministic' || s?.type !== 'sample' || z?.type !== 'sample') throw new Error('sites')
    const expected = toFlat(z.value).map((v, i) => i + Math.exp(item(s.value)) * v)
    for (const [i, v] of toFlat(x.value).entries()) expect(v).toBeCloseTo(expected[i] ?? Number.NaN, 13)
  })

  it('refuses observed sites, other supports and a centring outside [0, 1]', () => {
    const observed: Model = () => observe('x', normal(0, 1), 1)
    expect(() => traceOf(reparam(observed, { x: locScale(0) }), undefined)).toThrow(/observed/)
    // 'x' is observed there, so the handler is asked about an observed site only if the config names it.
    expect(() => locScale(2)).toThrow(/\[0, 1\]/)
    const halfLine: Model = () => sample('x', exp0())
    expect(() => traceOf(seed(reparam(halfLine, { x: locScale(0) }), makeKey(0)), undefined)).toThrow(
      /location and scale/,
    )
  })
})

function exp0(): ReturnType<typeof normal> {
  // A family without a location and a scale: the log-normal.
  return transformed(normal(0, 1), bijectors.exp()) as unknown as ReturnType<typeof normal>
}

describe('transformReparam', () => {
  const model: Model = () => {
    const s = sample('s', transformed(normal(0.2, 0.7), bijectors.exp()))
    observe('y', normal(s, 1), 1.3)
  }

  it('samples the base and applies the bijector', () => {
    const t = traceOf(seed(reparam(model, { s: transformReparam() }), makeKey(3)), undefined)
    expect([...t.sites.keys()]).toEqual(['s_base', 's', 'y'])
    const z = t.sites.get('s_base')
    const s = t.sites.get('s')
    if (z?.type !== 'sample' || s?.type !== 'deterministic') throw new Error('sites')
    expect(item(s.value)).toBeCloseTo(Math.exp(item(z.value)), 14)
  })

  it('has the density of the original program times the transform Jacobian', () => {
    const re = reparam(model, { s: transformReparam() })
    for (const z of [-1, 0.3, 1.2]) {
      const original = item(logJoint(model, undefined, { s: Math.exp(z) }).logJoint)
      const base = item(logJoint(re, undefined, { s_base: z }).logJoint)
      expect(base).toBeCloseTo(original + z, 12)
    }
  })

  it('rejects a distribution that is not transformed', () => {
    const plain: Model = () => sample('s', normal(0, 1))
    expect(() => traceOf(seed(reparam(plain, { s: transformReparam() }), makeKey(0)), undefined)).toThrow(/transformed/)
  })
})

describe('circular', () => {
  const model: Model = () => {
    const phi = sample('phi', vonMises(0.3, 2))
    observe('y', normal(phi, 1), 0.1)
  }

  it('is periodic and has the von Mises density at the wrapped angle', () => {
    const re = reparam(model, { phi: circular() })
    const base = item(logJoint(model, undefined, { phi: 0.5 }).logJoint)
    for (const turns of [0, 1, -2]) {
      const r = logJoint(re, undefined, { phi_unwrapped: 0.5 + turns * 2 * Math.PI })
      // Everything but the flat auxiliary site: the von Mises factor and the observation at the wrapped angle.
      expect(item(r.logJoint)).toBeCloseTo(base, 10)
    }
  })

  it('rejects other distributions', () => {
    const plain: Model = () => sample('phi', normal(0, 1))
    expect(() => traceOf(seed(reparam(plain, { phi: circular() }), makeKey(0)), undefined)).toThrow(/von Mises/)
  })
})

const draws = (
  post: { readonly draws?: Readonly<Record<string, { readonly data: ArrayLike<number> }>> },
  site: string,
): number[] => Array.from(post.draws?.[site]?.data ?? [])

describe('hierarchical funnels', () => {
  // Rubin's eight schools, centred: the group means are drawn around mu with spread tau.
  const Y = [28, 8, -3, 7, -1, 1, 18, 12]
  const SIGMA = [15, 10, 16, 11, 9, 11, 10, 18]
  const schools: Model = () => {
    const mu = sample('mu', normal(0, 5))
    const tau = sample('tau', halfCauchy(5))
    plate('J', 8, () => {
      const theta = sample('theta', normal(mu, tau))
      observe('y', normal(theta, tensor(SIGMA)), Y)
    })
  }

  it('diverges when centred, never when non-centred, and the non-centred chain reaches the neck', () => {
    const options = { key: makeKey(1), numChains: 1, numWarmup: 300, numDraws: 400 }
    const centred = mcmc(schools, undefined, options)
    const flat = mcmc(reparam(schools, { theta: locScale(0) }), undefined, options)
    expect(centred.diagnostics['divergences']).toBeGreaterThan(0)
    expect(flat.diagnostics['divergences']).toBe(0)
    // Small tau is the neck of the funnel: the centred chain does not get there.
    expect(Math.min(...draws(centred, 'tau'))).toBeGreaterThan(0.3)
    expect(Math.min(...draws(flat, 'tau'))).toBeLessThan(0.05)
  })

  it('reports the original site in the non-centred posterior', () => {
    const flat = mcmc(reparam(schools, { theta: locScale(0) }), undefined, {
      key: makeKey(2),
      numChains: 1,
      numWarmup: 100,
      numDraws: 20,
    })
    expect(flat.draws?.['theta']?.shape).toEqual([20, 8])
    expect(flat.draws?.['theta_decentered']?.shape).toEqual([20, 8])
  })
})
