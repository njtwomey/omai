import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../testing/arbitrary.ts'
import { key, uniform } from '../random/index.ts'
import { sampleChain } from './chains.ts'
import { blockedGibbs } from './gibbs.ts'

/** The exact transition matrix of updating variable `v` of a joint table pi[a][b] from its conditional. */
function conditionalUpdate(pi: readonly (readonly number[])[], v: 0 | 1): number[][] {
  const ka = pi.length
  const kb = pi[0]?.length ?? 0
  const states = ka * kb
  return Array.from({ length: states }, (_, s) => {
    const a = Math.floor(s / kb)
    const b = s % kb
    const row = new Array<number>(states).fill(0)
    if (v === 0) {
      const z = pi.reduce((acc, r) => acc + (r[b] ?? 0), 0)
      for (let a2 = 0; a2 < ka; a2++) row[a2 * kb + b] = (pi[a2]?.[b] ?? 0) / z
    } else {
      const z = (pi[a] ?? []).reduce((acc, x) => acc + x, 0)
      for (let b2 = 0; b2 < kb; b2++) row[a * kb + b2] = (pi[a]?.[b2] ?? 0) / z
    }
    return row
  })
}

function multiply(p: readonly (readonly number[])[], q: readonly (readonly number[])[]): number[][] {
  return p.map((r) => q[0]?.map((_, j) => r.reduce((acc, x, k) => acc + x * (q[k]?.[j] ?? 0), 0)) ?? [])
}

const tables = fc.tuple(fc.integer({ min: 2, max: 3 }), fc.integer({ min: 2, max: 3 })).chain(([ka, kb]) =>
  fc.array(fc.array(fc.double({ min: 0.05, max: 2, noNaN: true }), { minLength: kb, maxLength: kb }), {
    minLength: ka,
    maxLength: ka,
  }),
)

describe('Gibbs sampling on small discrete joints', () => {
  it('leaves the joint stationary: a systematic sweep satisfies pi P = pi', () => {
    fc.assert(
      fc.property(tables, (raw) => {
        const total = raw.flat().reduce((a, b) => a + b, 0)
        const pi = raw.map((r) => r.map((x) => x / total))
        const p = multiply(conditionalUpdate(pi, 0), conditionalUpdate(pi, 1))
        const flat = pi.flat()
        flat.forEach((target, j) => {
          const s = flat.reduce((acc, x, i) => acc + x * (p[i]?.[j] ?? 0), 0)
          expect(s).toBeCloseTo(target, 12)
        })
      }),
      { numRuns: RUNS },
    )
  })

  it('satisfies detailed balance for each single-variable update', () => {
    fc.assert(
      fc.property(tables, fc.constantFrom(0 as const, 1 as const), (raw, v) => {
        const total = raw.flat().reduce((a, b) => a + b, 0)
        const flat = raw.flat().map((x) => x / total)
        const pi = raw.map((r) => r.map((x) => x / total))
        const p = conditionalUpdate(pi, v)
        for (let i = 0; i < flat.length; i++)
          for (let j = 0; j < flat.length; j++)
            expect((flat[i] ?? 0) * (p[i]?.[j] ?? 0)).toBeCloseTo((flat[j] ?? 0) * (p[j]?.[i] ?? 0), 12)
      }),
      { numRuns: RUNS },
    )
  })

  it('visits a 2 x 2 joint in proportion when run with blockedGibbs', () => {
    const pi = [
      [0.1, 0.2],
      [0.3, 0.4],
    ]
    const draw = (u: number, w: readonly number[]): number => (u < (w[0] ?? 0) / ((w[0] ?? 0) + (w[1] ?? 0)) ? 0 : 1)
    const k = blockedGibbs(2, [
      {
        indices: [0],
        sample: (kk, x) => [draw(uniform(kk, [1]).data[0] ?? 0, [pi[0]?.[x[1] ?? 0] ?? 0, pi[1]?.[x[1] ?? 0] ?? 0])],
      },
      { indices: [1], sample: (kk, x) => [draw(uniform(kk, [1]).data[0] ?? 0, pi[x[0] ?? 0] ?? [])] },
    ])
    const n = 10000
    const r = sampleChain(k, [0, 0], { key: key(0), numDraws: n })
    const counts = [0, 0, 0, 0]
    for (let i = 0; i < n; i++) {
      const s = 2 * (r.draws[2 * i] ?? 0) + (r.draws[2 * i + 1] ?? 0)
      counts[s] = (counts[s] ?? 0) + 1
    }
    pi.flat().forEach((p, s) => {
      expect(Math.abs((counts[s] ?? 0) / n - p)).toBeLessThan(0.025)
    })
  })
})
