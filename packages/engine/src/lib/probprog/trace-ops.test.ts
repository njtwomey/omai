// Trace operations: assess is the log joint, generate and regenerate carry the weights Metropolis-Hastings needs
// (checked against a direct computation and, for regenerate, by the invariance of the posterior under the move).
import { describe, expect, it } from 'vitest'
import { normal } from '../dists/index.ts'
import { normal as drawNormal, key as makeKey, uniform } from '../random/index.ts'
import { item, type Value } from '../tensor/index.ts'
import { logJoint, siteLogProbs } from './density.ts'
import { seed, traceOf } from './handlers.ts'
import { type Model, observe, plate, sample } from './primitives.ts'
import { assess, generate, inSelection, latentSites, latentValues, propose, regenerate, update } from './trace-ops.ts'

// A three-level hierarchy with an observation under each leaf.
const HIER: Model = () => {
  const m = sample('m', normal(0, 2))
  const a = sample('a', normal(m, 1))
  const b = sample('b', normal(m, 1))
  observe('ya', normal(a, 0.5), 1.1)
  observe('yb', normal(b, 0.5), -0.4)
}

const num = (v: Value | undefined): number => item(v as Value)
const lp = (t: ReturnType<typeof traceOf>, name: string): number => num(siteLogProbs(t)[name])

describe('assess, propose, generate', () => {
  it('assess equals the log joint at the same values', () => {
    const values = { m: 0.3, a: -0.2, b: 0.9 }
    expect(assess(HIER, undefined, values)).toBeCloseTo(item(logJoint(HIER, undefined, values).logJoint), 14)
  })

  it('assess needs every latent site', () => {
    expect(() => assess(HIER, undefined, { m: 0 })).toThrow(/key/)
  })

  it('propose scores the latent draws given their parents', () => {
    const { trace, logWeight } = propose(HIER, undefined, makeKey(5))
    expect(logWeight).toBeCloseTo(lp(trace, 'm') + lp(trace, 'a') + lp(trace, 'b'), 14)
  })

  it('generate: the weight is the constrained and observed sites given the rest', () => {
    const none = generate(HIER, undefined, makeKey(1), {})
    expect(none.logWeight).toBeCloseTo(lp(none.trace, 'ya') + lp(none.trace, 'yb'), 14)
    const some = generate(HIER, undefined, makeKey(1), { m: 0.5 })
    expect(some.logWeight).toBeCloseTo(lp(some.trace, 'm') + lp(some.trace, 'ya') + lp(some.trace, 'yb'), 14)
    expect(num(latentValues(some.trace)['m'])).toBe(0.5)
    const full = generate(HIER, undefined, makeKey(1), { m: 0.5, a: 0.1, b: -0.1 })
    expect(full.logWeight).toBeCloseTo(assess(HIER, undefined, { m: 0.5, a: 0.1, b: -0.1 }), 14)
  })
})

describe('regenerate', () => {
  const start = traceOf(seed(HIER, makeKey(3)), undefined)

  it('keeps unselected sites, redraws the selected, and returns the discarded values', () => {
    const r = regenerate(HIER, undefined, start, ['a'], makeKey(8))
    const before = latentValues(start)
    const after = latentValues(r.trace)
    expect(num(after['m'])).toBe(num(before['m']))
    expect(num(after['b'])).toBe(num(before['b']))
    expect(num(after['a'])).not.toBe(num(before['a']))
    expect(Object.keys(r.discarded)).toEqual(['a'])
    expect(num(r.discarded['a'])).toBe(num(before['a']))
  })

  it('weight: new minus old log density of the sites that were kept', () => {
    const r = regenerate(HIER, undefined, start, ['a'], makeKey(8))
    // Redrawing a leaves m, b, ya (through a) and yb: the terms of a and ya move, a's own prior cancels.
    const expected = lp(r.trace, 'ya') - lp(start, 'ya')
    expect(r.logWeight).toBeCloseTo(expected, 13)
    // Redrawing the root m changes the priors of both leaves, which are kept.
    const root = regenerate(HIER, undefined, start, ['m'], makeKey(9))
    expect(root.logWeight).toBeCloseTo(lp(root.trace, 'a') + lp(root.trace, 'b') - lp(start, 'a') - lp(start, 'b'), 13)
  })

  it('selection by predicate, and an empty selection has weight zero', () => {
    const r = regenerate(HIER, undefined, start, (s) => s.name === 'a' || s.name === 'b', makeKey(2))
    expect(Object.keys(r.discarded)).toEqual(['a', 'b'])
    expect(regenerate(HIER, undefined, start, [], makeKey(2)).logWeight).toBe(0)
    const site = latentSites(start)[0]
    expect(site !== undefined && inSelection((s) => s.name === 'm', site)).toBe(true)
  })

  it('leaves the target invariant: exact posterior draws pushed through a Metropolis step keep their law', () => {
    // The posterior of m given y = 1.2 under m ~ N(0, 1), y ~ N(m, 1) is N(0.6, 0.5).
    const one: Model = () => {
      const m = sample('m', normal(0, 1))
      observe('y', normal(m, 1), 1.2)
    }
    const n = 3000
    const moved: number[] = []
    for (let i = 0; i < n; i++) {
      const m0 = 0.6 + Math.sqrt(0.5) * item(drawNormal(makeKey(100 + i), []))
      const at = update(one, undefined, traceOf(seed(one, makeKey(i)), undefined), { m: m0 }).trace
      const proposal = regenerate(one, undefined, at, ['m'], makeKey(7000 + i))
      const accept = item(uniform(makeKey(9000 + i), [])) < Math.min(1, Math.exp(proposal.logWeight))
      moved.push(num(latentValues(accept ? proposal.trace : at)['m']))
    }
    const mean = moved.reduce((a, b) => a + b, 0) / n
    const sd = Math.sqrt(moved.reduce((a, b) => a + (b - mean) ** 2, 0) / n)
    expect(Math.abs(mean - 0.6)).toBeLessThan((4 * Math.sqrt(0.5)) / Math.sqrt(n))
    expect(Math.abs(sd - Math.sqrt(0.5))).toBeLessThan(0.05)
  })
})

describe('update', () => {
  const start = traceOf(seed(HIER, makeKey(3)), undefined)

  it('weight is the new minus the old assessed density', () => {
    const before = latentValues(start)
    const u = update(HIER, undefined, start, { a: 0.25 })
    const old = Object.fromEntries(Object.entries(before).map(([k, v]) => [k, num(v)]))
    expect(u.logWeight).toBeCloseTo(assess(HIER, undefined, { ...old, a: 0.25 }) - assess(HIER, undefined, old), 13)
    expect(num(latentValues(u.trace)['a'])).toBe(0.25)
    expect(num(u.discarded['a'])).toBe(old['a'])
  })

  it('a change to an observed site is a new observation', () => {
    const u = update(HIER, undefined, start, { ya: 3 })
    expect(num((u.trace.sites.get('ya') as { value: Value }).value)).toBe(3)
    expect(u.logWeight).toBeCloseTo(lp(u.trace, 'ya') - lp(start, 'ya'), 13)
  })

  it('rejects a name the program does not have', () => {
    expect(() => update(HIER, undefined, start, { nope: 1 })).toThrow(/no site nope/)
  })

  it('structure that changes with a value: new latent sites are drawn, and count for nothing in the weight', () => {
    const branching: Model = () => {
      const z = sample('z', normal(0, 1))
      if (num(z) > 0) sample('extra', normal(0, 1))
      observe('y', normal(z, 1), 0.3)
    }
    const t = traceOf(seed(branching, makeKey(0)), undefined)
    const down = update(branching, undefined, t, { z: -1 }, makeKey(1))
    expect(down.trace.sites.has('extra')).toBe(false)
    const up = update(branching, undefined, down.trace, { z: 1 }, makeKey(2))
    expect(up.trace.sites.has('extra')).toBe(true)
    expect(up.logWeight).toBeCloseTo(
      assess(branching, undefined, { z: 1, extra: num(latentValues(up.trace)['extra']) }) -
        lp(up.trace, 'extra') -
        assess(branching, undefined, { z: -1 }),
      13,
    )
  })

  it('works inside plates and keeps their subsamples', () => {
    const plated: Model = () => {
      const m = sample('m', normal(0, 1))
      plate('n', 3, () => observe('y', normal(m, 1), [0.1, 0.2, 0.3]))
    }
    const t = traceOf(seed(plated, makeKey(0)), undefined)
    const u = update(plated, undefined, t, { m: 0.4 })
    expect(u.logWeight).toBeCloseTo(
      assess(plated, undefined, { m: 0.4 }) - assess(plated, undefined, { m: num(latentValues(t)['m']) }),
      13,
    )
  })
})
