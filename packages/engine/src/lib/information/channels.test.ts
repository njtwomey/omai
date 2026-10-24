import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../testing/arbitrary.ts'
import { checkPrimitive } from '../autodiff/index.ts'
import { item, tensor, toFlat, type Value } from '../tensor/index.ts'
import { recorder } from '../trace/index.ts'
import * as I from './index.ts'

const row = (n: number): fc.Arbitrary<number[]> =>
  fc.array(fc.double({ min: 0.01, max: 1, noNaN: true }), { minLength: n, maxLength: n }).map((w) => {
    const s = w.reduce((a, b) => a + b, 0)
    return w.map((v) => v / s)
  })
const channel = (nx: number, ny: number): fc.Arbitrary<number[][]> =>
  fc.array(row(ny), { minLength: nx, maxLength: nx })
const compose = (a: readonly (readonly number[])[], b: readonly (readonly number[])[]): number[][] =>
  a.map((r) => (b[0] ?? []).map((_, j) => r.reduce((s, v, k) => s + v * (b[k]?.[j] ?? 0), 0)))

describe('channels', () => {
  it('Blahut-Arimoto agrees with the closed-form capacities', () => {
    for (const p of [0.01, 0.1, 0.3, 0.5]) {
      const bsc = I.binarySymmetricChannel(p, { base: 2 })
      expect(I.channelCapacity(bsc.transition, { base: 2 }).capacity).toBeCloseTo(bsc.capacity, 10)
      const bec = I.binaryErasureChannel(p, { base: 2 })
      expect(I.channelCapacity(bec.transition, { base: 2 }).capacity).toBeCloseTo(bec.capacity, 10)
      const z = I.zChannel(p, { base: 2 })
      expect(I.channelCapacity(z.transition, { base: 2 }).capacity).toBeCloseTo(z.capacity, 10)
    }
    expect(I.zChannel(1).capacity).toBe(0)
    const result = I.channelCapacity(I.zChannel(0.5).transition)
    // The optimal input of the Z channel sends a 1 with probability 2/5 when p = 1/2.
    expect(toFlat(result.input)[1]).toBeCloseTo(0.4, 8)
    expect(result.lower).toBeLessThanOrEqual(result.upper)
    expect(() => I.binarySymmetricChannel(1.5)).toThrow()
    expect(() => I.channelCapacity([[0.5, 0.4]])).toThrow()
    expect(() => I.channelCapacity([1, 0])).toThrow()
  })

  it('records the bounds closing in', () => {
    const rec = recorder()
    I.channelCapacity(I.zChannel(0.3).transition, { tracer: rec })
    const gaps = rec.events
      .filter((e) => e.type === 'metric' && e.name === 'gap')
      .map((e) => (e as { value: number }).value)
    expect(gaps.length).toBeGreaterThan(2)
    expect(gaps[gaps.length - 1]).toBeLessThan(1e-10)
    for (let i = 1; i < gaps.length; i++) expect(gaps[i] ?? 0).toBeLessThanOrEqual((gaps[i - 1] ?? 0) + 1e-15)
    // An exhausted budget still returns bounds that bracket the capacity.
    const early = I.channelCapacity(I.zChannel(0.3).transition, { maxIter: 2 })
    expect(early.converged).toBe(false)
    expect(early.lower).toBeLessThanOrEqual(I.zChannel(0.3).capacity + 1e-12)
    expect(early.upper).toBeGreaterThanOrEqual(I.zChannel(0.3).capacity - 1e-12)
    expect(() =>
      I.blahutArimotoAlgorithm(
        [
          [1, 0],
          [0, 1],
        ],
        { init: [1] },
      ),
    ).toThrow()
  })

  it('capacity bounds every input and is reduced by processing (property)', () => {
    fc.assert(
      fc.property(channel(3, 4), channel(4, 3), row(3), (w1, w2, p) => {
        // Bounds, not point estimates: they hold whether or not the run converged.
        const c1 = I.channelCapacity(w1, { maxIter: 500 })
        expect(item(I.channelMutualInformation(p, w1))).toBeLessThanOrEqual(c1.upper + 1e-12)
        expect(c1.lower).toBeLessThanOrEqual(Math.log(3) + 1e-12)
        const c12 = I.channelCapacity(compose(w1, w2), { maxIter: 500 })
        const c2 = I.channelCapacity(w2, { maxIter: 500 })
        expect(c12.lower).toBeLessThanOrEqual(Math.min(c1.upper, c2.upper) + 1e-12)
        expect(c1.converged && c2.converged && c12.converged).toBe(true)
      }),
      { numRuns: RUNS },
    )
  })

  it('converges quickly when the optimal input leaves symbols unused or rows repeat (regression)', () => {
    // These channels stalled plain Blahut-Arimoto for over 100000 steps: the optimum is on (or next to) the boundary
    // of the simplex, where the multiplicative update converges sublinearly.
    const nearlyUseless = [
      [0.2533411830670193, 0.6592230217879231, 0.0874357951450576],
      [0.24065675309383536, 0.6817777038697097, 0.07756554303645509],
      [0.2507049317934575, 0.6774583452491597, 0.07183672295738293],
    ]
    const smallSupport = [
      [0.0057930535359044895, 0.5502242849990292, 0.2640283455146213, 0.179954315950445],
      [0.008354095507253405, 0.009769439338443764, 0.835409550725338, 0.146466914428965],
      [0.012846186199997512, 0.02902056708149005, 0.9209211843468392, 0.037212062371673224],
    ]
    const repeated = [
      [0.6, 0.4],
      [0.6, 0.4],
      [0.1, 0.9],
      [0.35, 0.65],
    ]
    for (const w of [
      nearlyUseless,
      smallSupport,
      repeated,
      [[1, 0]],
      [
        [0.3, 0.7],
        [0.3, 0.7],
      ],
    ]) {
      const result = I.channelCapacity(w, { tol: 1e-12 })
      expect(result.converged).toBe(true)
      expect(result.iterations).toBeLessThan(500)
      expect(result.upper - result.lower).toBeLessThan(1e-12)
    }
    expect(I.channelCapacity(nearlyUseless).capacity).toBeCloseTo(0.0004482794292515, 15)
  })

  it('channel mutual information is differentiable', () => {
    const f = (p: Value, w: Value): Value => I.channelMutualInformation(p, w)
    const w = tensor([
      [0.7, 0.2, 0.1],
      [0.1, 0.3, 0.6],
    ])
    expect(checkPrimitive(f, [tensor([0.4, 0.6]), w], { vmap: false }).failures).toEqual([])
    expect(() => I.channelMutualInformation([0.5, 0.5], [[1, 0]])).toThrow()
  })

  it('water-filling satisfies its optimality conditions', () => {
    fc.assert(
      fc.property(
        fc.array(fc.double({ min: 0.1, max: 5, noNaN: true }), { minLength: 1, maxLength: 8 }),
        fc.double({ min: 0, max: 20, noNaN: true }),
        (noise, power) => {
          const { allocation, level, total } = I.waterFilling(noise, power)
          expect(allocation.reduce((a, b) => a + b, 0)).toBeCloseTo(power, 9)
          noise.forEach((n, i) => {
            const a = allocation[i] ?? 0
            if (a > 0) expect(a + n).toBeCloseTo(level, 9)
            else expect(n).toBeGreaterThanOrEqual(level - 1e-9)
          })
          // Any other split (here: proportional to 1 / N) does no better.
          const inv = noise.map((n) => 1 / n)
          const s = inv.reduce((a, b) => a + b, 0)
          const other = noise.reduce((acc, n, i) => acc + 0.5 * Math.log1p((((inv[i] ?? 0) / s) * power) / n), 0)
          expect(total).toBeGreaterThanOrEqual(other - 1e-9)
        },
      ),
      { numRuns: RUNS },
    )
    expect(() => I.waterFilling([], 1)).toThrow()
  })
})
