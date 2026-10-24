import { describe, expect, it } from 'vitest'
import { normal } from '../dists/index.ts'
import { key } from '../random/index.ts'
import { item, tensor, toFlat, type Value } from '../tensor/index.ts'
import {
  block,
  deterministic,
  doIntervention,
  factor,
  handle,
  logJoint,
  type Model,
  markov,
  observe,
  param,
  plate,
  replay,
  sample,
  seed,
  substitute,
  traceOf,
  uncondition,
} from './index.ts'

describe('primitives', () => {
  it('needs a key to draw, and none to observe', () => {
    expect(() => sample('x', normal(0, 1))).toThrow(/seed/)
    expect(item(sample('y', normal(0, 1), { obs: 2 }))).toBe(2)
  })

  it('records deterministic sites, params and factors', () => {
    const model: Model = () => {
      const a = param('a', 3, { support: { kind: 'positive' } })
      deterministic('twice', [item(a) * 2])
      factor('f', -1.5)
    }
    const t = traceOf(model, undefined)
    expect([...t.sites.keys()]).toEqual(['a', 'twice', 'f'])
    const f = t.sites.get('f')
    expect(f?.type === 'sample' && f.isFactor && f.infer.isAuxiliary).toBe(true)
    expect(item(logJoint(model, undefined).logJoint)).toBe(-1.5)
    const a = traceOf(substitute(model, { a: 5 }), undefined).sites.get('a')
    expect(a?.type === 'param' ? item(a.value) : 0).toBe(5)
  })

  it('rejects duplicate site names', () => {
    const model: Model = () => {
      observe('y', normal(0, 1), 0)
      observe('y', normal(0, 1), 1)
    }
    expect(() => traceOf(model, undefined)).toThrow(/twice/)
  })

  it('subsamples plates and slices params inside them', () => {
    const model: Model = () => {
      plate(
        'data',
        6,
        (idx) => {
          const loc = param('loc', tensor([0, 1, 2, 3, 4, 5]))
          observe(
            'y',
            normal(loc, 1),
            toFlat(idx).map((i) => i),
          )
        },
        { subsampleSize: 2 },
      )
    }
    const t = traceOf(seed(model, key(4)), undefined)
    const p = t.sites.get('data')
    const y = t.sites.get('y')
    const loc = t.sites.get('loc')
    expect(p?.type === 'plate' ? p.value.shape : []).toEqual([2])
    // The local parameter is sliced to the subsample, and the observations are scaled by 6 / 2.
    expect(toFlat(loc?.type === 'param' ? loc.value : tensor([]))).toEqual(
      toFlat(p?.type === 'plate' ? p.value : tensor([])),
    )
    expect(y?.type === 'sample' ? y.scale : 0).toBe(3)
  })

  it('marks markov steps and runs loops of any length', () => {
    const walk: Model = () => {
      markov('t', 3, (t, x: Value | number) => sample(`x${t}`, normal(x, 1)), 0, { history: 2 })
    }
    const t = traceOf(seed(walk, key(0)), undefined)
    const s = t.sites.get('x2')
    expect(s?.type === 'sample' ? s.markov[0] : undefined).toEqual({ name: 't', step: 2, history: 2 })
    const loop = t.sites.get('t')
    expect(loop?.type === 'markov' ? [loop.length, loop.history] : []).toEqual([3, 2])
    expect(() => markov('bad', -1, (_, c: number) => c, 0)).toThrow(/length/)
  })
})

describe('handlers', () => {
  const model: Model = () => {
    const mu = sample('mu', normal(0, 1))
    const z = sample('z', normal(mu, 1))
    observe('y', normal(z, 1), 0.5)
  }

  it('replays a trace and blocks by type', () => {
    const first = traceOf(seed(model, key(1)), undefined)
    const again = traceOf(replay(model, first), undefined)
    expect(
      item(again.sites.get('z')?.type === 'sample' ? (again.sites.get('z') as { value: Value }).value : tensor(0)),
    ).toBe(item(first.sites.get('z')?.type === 'sample' ? (first.sites.get('z') as { value: Value }).value : tensor(1)))
    const visible = [...traceOf(block(seed(model, key(1)), { exposeTypes: ['sample'] }), undefined).sites.keys()]
    expect(visible).toEqual(['mu', 'z', 'y'])
    const hidden = [
      ...traceOf(block(seed(model, key(1)), { hideFn: (m) => m.name.startsWith('m') }), undefined).sites.keys(),
    ]
    expect(hidden).toEqual(['z', 'y'])
  })

  it('unconditions observations to draw replicated data', () => {
    const t = traceOf(seed(uncondition(model), key(2)), undefined)
    const y = t.sites.get('y')
    expect(y?.type === 'sample' && !y.isObserved && item(y.value) !== 0.5).toBe(true)
  })

  it('intervenes downstream while recording the original draw', () => {
    const t = traceOf(seed(doIntervention(model, { z: 4 }), key(3)), undefined)
    const z = t.sites.get('z')
    expect(z?.type === 'sample' && !z.isObserved).toBe(true)
    expect(z?.type === 'sample' ? item(z.value) : 4).not.toBe(4)
    expect([...t.sites.keys()]).toEqual(['mu', 'z', 'y'])
  })

  it('runs custom handlers', () => {
    const doubled = handle(model, () => ({
      process: (m) => {
        if (m.type === 'sample' && !m.isObserved) m.value = tensor(2)
      },
    }))
    const t = traceOf(doubled, undefined)
    expect(item(t.sites.get('mu')?.type === 'sample' ? (t.sites.get('mu') as { value: Value }).value : tensor(0))).toBe(
      2,
    )
  })
})
