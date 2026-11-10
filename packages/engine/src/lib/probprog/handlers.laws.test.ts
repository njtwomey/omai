// Laws of the handler layer: how handlers compose, what block hides, what replay and condition fix, that the same
// seed gives the same program, and the added handlers (scope, lift, mean, inferHint, inferConfig).
import { describe, expect, it } from 'vitest'
import { exponential, normal } from '../dists/index.ts'
import { key as makeKey } from '../random/index.ts'
import { item, type Value } from '../tensor/index.ts'
import { logJoint } from './density.ts'
import {
  block,
  condition,
  inferConfig,
  inferHint,
  mean,
  replay,
  scale,
  scope,
  seed,
  substitute,
  traceOf,
} from './handlers.ts'
import { lift } from './lift.ts'
import { type Model, observe, param, plate, sample } from './primitives.ts'

const CHAIN: Model = () => {
  const a = sample('a', normal(0, 1))
  const b = sample('b', normal(a, 1))
  const c = sample('c', normal(b, 2))
  observe('y', normal(c, 0.5), 0.8)
}

const valueAt = (t: ReturnType<typeof traceOf>, name: string): number => {
  const s = t.sites.get(name)
  return s === undefined || s.type === 'param' || s.type === 'plate' || s.type === 'markov' ? Number.NaN : item(s.value)
}
const lj = (m: Model, values: Record<string, number> = {}): number => item(logJoint(m, undefined, values).logJoint)

describe('composition laws', () => {
  it('seed: the same key gives the same program, and a different key another one', () => {
    const k = makeKey(11)
    const a = traceOf(seed(CHAIN, k), undefined)
    const b = traceOf(seed(CHAIN, k), undefined)
    const c = traceOf(seed(CHAIN, makeKey(12)), undefined)
    expect(['a', 'b', 'c'].map((n) => valueAt(a, n))).toEqual(['a', 'b', 'c'].map((n) => valueAt(b, n)))
    expect(valueAt(a, 'a')).not.toBe(valueAt(c, 'a'))
  })

  it('seed: observed sites consume no keys, so adding an observation leaves the draws alone', () => {
    const plain: Model = () => {
      const a = sample('a', normal(0, 1))
      sample('b', normal(a, 1))
    }
    const more: Model = () => {
      const a = sample('a', normal(0, 1))
      observe('o', normal(a, 1), 0.2)
      sample('b', normal(a, 1))
    }
    const k = makeKey(4)
    const p = traceOf(seed(plain, k), undefined)
    const m = traceOf(seed(more, k), undefined)
    expect(valueAt(m, 'a')).toBe(valueAt(p, 'a'))
    expect(valueAt(m, 'b')).toBe(valueAt(p, 'b'))
  })

  it('substitute composes: nesting equals one call with both sets of values', () => {
    const nested = substitute(substitute(CHAIN, { a: 0.5 }), { b: -1, c: 0.2 })
    const flat = substitute(CHAIN, { a: 0.5, b: -1, c: 0.2 })
    expect(lj(nested)).toBe(lj(flat))
    // The outer handler wins on a shared name, as the innermost handler is the first to act and the outer one overwrites.
    const clash = substitute(substitute(CHAIN, { a: 0.5, b: 0, c: 0 }), { a: 1 })
    expect(lj(clash)).toBe(lj(CHAIN, { a: 1, b: 0, c: 0 }))
  })

  it('condition: observing a site gives the density of substituting it', () => {
    const conditioned = condition(CHAIN, { a: 0.3, b: 0.1 })
    expect(lj(conditioned, { c: 0.4 })).toBeCloseTo(lj(CHAIN, { a: 0.3, b: 0.1, c: 0.4 }), 14)
    expect(traceOf(seed(conditioned, makeKey(0)), undefined).sites.get('a')).toBeDefined()
  })

  it('scale composes multiplicatively', () => {
    expect(lj(scale(scale(CHAIN, 2), 3), { a: 0, b: 0, c: 0 })).toBeCloseTo(6 * lj(CHAIN, { a: 0, b: 0, c: 0 }), 12)
  })

  it('block hides sites from outer handlers without changing any value', () => {
    const k = makeKey(2)
    const open = traceOf(seed(CHAIN, k), undefined)
    const blocked = traceOf(block(seed(CHAIN, k), { hide: ['a', 'y'] }), undefined)
    expect([...blocked.sites.keys()]).toEqual(['b', 'c'])
    expect(valueAt(blocked, 'b')).toBe(valueAt(open, 'b'))
    expect(valueAt(blocked, 'c')).toBe(valueAt(open, 'c'))
    // An outer condition cannot reach a hidden site.
    const fixed = traceOf(condition(block(seed(CHAIN, k), { hide: ['a'] }), { a: 100 }), undefined)
    expect(Math.abs(valueAt(fixed, 'b'))).toBeLessThan(50)
  })

  it('replay reproduces a trace without a key and is idempotent', () => {
    const first = traceOf(seed(CHAIN, makeKey(9)), undefined)
    const again = traceOf(replay(CHAIN, first), undefined)
    const twice = traceOf(replay(replay(CHAIN, first), first), undefined)
    for (const n of ['a', 'b', 'c']) {
      expect(valueAt(again, n)).toBe(valueAt(first, n))
      expect(valueAt(twice, n)).toBe(valueAt(first, n))
    }
  })

  it('replay and condition agree on the log joint of the sites they fix', () => {
    const t = traceOf(seed(CHAIN, makeKey(6)), undefined)
    const values = { a: valueAt(t, 'a'), b: valueAt(t, 'b'), c: valueAt(t, 'c') }
    expect(item(logJoint(replay(CHAIN, t), undefined, {}).logJoint)).toBeCloseTo(lj(CHAIN, values), 13)
  })
})

describe('scope', () => {
  const part: Model = () => {
    const theta = sample('theta', normal(0, 1))
    plate('n', 2, () => observe('y', normal(theta, 1), [0.1, 0.2]))
  }

  it('prefixes the names, nests, and is undone by conditioning on the prefixed name', () => {
    const both: Model = () => {
      scope(part, 'a')(undefined)
      scope(scope(part, 'b'), 'outer')(undefined)
    }
    const t = traceOf(seed(both, makeKey(0)), undefined)
    expect([...t.sites.keys()].filter((n) => n.endsWith('theta'))).toEqual(['a/theta', 'outer/b/theta'])
    const y = t.sites.get('a/y')
    expect(y?.type === 'sample' && y.plates.map((p) => p.name)).toEqual(['a/n'])
    // Names round-trip: values given under the prefixed names are the ones the sub-model uses.
    const fixed = lj(scope(part, 'p'), { 'p/theta': 0.7 })
    expect(fixed).toBeCloseTo(lj(part, { theta: 0.7 }), 14)
  })

  it('leaves the types it is told to hide', () => {
    const t = traceOf(seed(scope(part, 's', { divider: '.', hideTypes: ['plate'] }), makeKey(0)), undefined)
    expect([...t.sites.keys()]).toContain('s.theta')
    const y = t.sites.get('s.y')
    expect(y?.type === 'sample' && y.plates.map((p) => p.name)).toEqual(['n'])
  })
})

describe('lift', () => {
  const model: Model = () => {
    const s = param('s', 1, { support: { kind: 'positive' } })
    observe('y', normal(0, s), 0.5)
  }

  it('turns a param into a sample site with the prior, so the density gains the prior', () => {
    const lifted = lift(model, { s: exponential(2) })
    expect(lj(lifted, { s: 0.8 })).toBeCloseTo(lj(model, { s: 0.8 }) + Math.log(2) - 2 * 0.8, 13)
    const site = traceOf(seed(lifted, makeKey(1)), undefined).sites.get('s')
    expect(site?.type).toBe('sample')
  })

  it('takes one prior for every param, leaves params without a prior, and shares repeated calls', () => {
    const two: Model = () => {
      const a = param('a', 1)
      const b = param('b', 2)
      const a2 = param('a', 1)
      observe('y', normal(a, 1), 0)
      return [a, b, a2]
    }
    const t = traceOf(seed(lift(two, { a: normal(0, 1) }), makeKey(3)), undefined)
    expect(t.sites.get('b')?.type).toBe('param')
    const out = t.returnValue as Value[]
    expect(item(out[0] as Value)).toBe(item(out[2] as Value))
    const all = traceOf(seed(lift(two, normal(0, 1)), makeKey(3)), undefined)
    expect(all.sites.get('b')?.type).toBe('sample')
  })
})

describe('mean, inferHint, inferConfig', () => {
  it('mean runs the program at its expected values without a key', () => {
    const t = traceOf(mean(CHAIN), undefined)
    expect(valueAt(t, 'a')).toBe(0)
    expect(valueAt(t, 'b')).toBe(0)
    expect(valueAt(t, 'c')).toBe(0)
  })

  it('inferHint annotates the named sites and nothing else', () => {
    const t = traceOf(seed(inferHint(CHAIN, ['a', 'b'], { method: 'gibbs', block: 'top' }), makeKey(0)), undefined)
    const infer = (n: string) => {
      const s = t.sites.get(n)
      return s?.type === 'sample' ? s.infer : undefined
    }
    expect(infer('a')).toMatchObject({ method: 'gibbs', block: 'top' })
    expect(infer('c')?.method).toBeUndefined()
  })

  it('inferConfig merges a function of the message', () => {
    const t = traceOf(
      seed(
        inferConfig(CHAIN, (m) => (m.isObserved ? undefined : { block: m.name })),
        makeKey(0),
      ),
      undefined,
    )
    const b = t.sites.get('b')
    expect(b?.type === 'sample' && b.infer.block).toBe('b')
  })
})
