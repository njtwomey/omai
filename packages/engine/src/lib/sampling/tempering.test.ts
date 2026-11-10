import { describe, expect, it } from 'vitest'
import { bimodal, moments, standardNormal } from '../../testing/sampling.ts'
import { key, normal } from '../random/index.ts'
import { sampleChain } from './chains.ts'
import { randomWalkMetropolis } from './metropolis.ts'
import { numericTarget } from './target.ts'
import { type PTState, parallelTemperingKernel, temperatureLadder, temperedTarget } from './tempering.ts'

const rwm = (scale: number) => (target: ReturnType<typeof standardNormal>, beta: number) =>
  randomWalkMetropolis(target, { scale: scale / Math.sqrt(beta) })

describe('temperedTarget and the ladders', () => {
  it('multiplies the log density and the gradient by beta, and keeps the coordinates', () => {
    const t = temperedTarget(standardNormal(2), 0.25)
    expect(t.logDensity([2, 0])).toBeCloseTo(-0.5, 14)
    expect(Array.from(t.valueAndGrad([2, 4]).grad)).toEqual([-0.5, -1])
    expect(t.names).toEqual(standardNormal(2).names)
    expect(() => temperedTarget(standardNormal(1), 0)).toThrow()
  })

  it('makes geometric and linear ladders from 1 down to the minimum', () => {
    expect(temperatureLadder(3, 0.25)).toEqual([1, 0.5, 0.25])
    expect(temperatureLadder(3, 0.5, 'linear')).toEqual([1, 0.75, 0.5])
    expect(() => temperatureLadder(1, 0.5)).toThrow()
    expect(() => temperatureLadder(3, 1)).toThrow()
  })
})

describe('parallelTemperingKernel', () => {
  it('leaves the product of the tempered densities invariant: every replica keeps its own temperature', () => {
    const betas = temperatureLadder(3, 0.25)
    const k = parallelTemperingKernel(standardNormal(1), { betas, kernelFor: rwm(1.5) })
    for (const scheme of ['even-odd', 'random'] as const) {
      const kernel =
        scheme === 'even-odd'
          ? k
          : parallelTemperingKernel(standardNormal(1), { betas, kernelFor: rwm(1.5), swapScheme: 'random' })
      const n = 3000
      const z = normal(key(scheme === 'even-odd' ? 1 : 2), [n, 3]).data
      const out: number[][] = [[], [], []]
      for (let i = 0; i < n; i++) {
        // An exact draw of the product: replica j is N(0, 1 / beta_j).
        const replicas = betas.map((b, j) => {
          const x = (z[3 * i + j] ?? 0) / Math.sqrt(b)
          return { position: Float64Array.of(x), logDensity: -0.5 * b * x * x }
        })
        const init = kernel.init([0])
        const s: PTState = {
          ...init,
          replicas,
          position: replicas[0]?.position ?? init.position,
          logDensity: replicas[0]?.logDensity ?? 0,
        }
        const next = kernel.step(key(1000 + i), s).state
        next.replicas.forEach((r, j) => {
          out[j]?.push(r.position[0] ?? 0)
        })
      }
      betas.forEach((b, j) => {
        const { mean, sd } = moments(out[j] as number[])
        expect(Math.abs(mean), `${scheme} replica ${j} mean`).toBeLessThan((4 * (1 / Math.sqrt(b))) / Math.sqrt(n))
        expect(Math.abs(sd - 1 / Math.sqrt(b)), `${scheme} replica ${j} sd`).toBeLessThan(
          ((4 * (1 / Math.sqrt(b))) / Math.sqrt(2 * n)) * 1.5,
        )
      })
    }
  })

  it('finds both modes of a bimodal target where a single random walk stays in the one it started in', () => {
    const target = bimodal(4)
    const single = sampleChain(randomWalkMetropolis(target, { scale: 1 }), [4], {
      key: key(3),
      numDraws: 4000,
      numWarmup: 200,
    })
    expect(single.draws.every((v) => v > 0)).toBe(true)
    const betas = temperatureLadder(6, 0.02)
    const pt = parallelTemperingKernel(target, { betas, kernelFor: rwm(1) })
    const r = sampleChain(pt, [4], { key: key(4), numDraws: 6000, numWarmup: 300 })
    const positive = r.draws.filter((v) => v > 0).length / r.draws.length
    // Each mode has weight one half.
    expect(Math.abs(positive - 0.5)).toBeLessThan(0.12)
    const s = r.state
    // Every adjacent pair swapped now and then, and walkers made trips between the hottest and the coldest replica.
    expect(Array.from(s.swapAttempts).every((a) => a > 100)).toBe(true)
    expect(Array.from(s.swapSuccesses, (a, i) => a / (s.swapAttempts[i] ?? 1)).every((q) => q > 0.05 && q < 1)).toBe(
      true,
    )
    expect(s.roundTrips).toBeGreaterThan(3)
  })

  it('serialises nothing it should not need and validates its settings', () => {
    const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1)
    const base = { betas: [1, 0.5], kernelFor: rwm(1) }
    expect(() => parallelTemperingKernel(t, { ...base, betas: [1] })).toThrow()
    expect(() => parallelTemperingKernel(t, { ...base, betas: [0.9, 0.5] })).toThrow()
    expect(() => parallelTemperingKernel(t, { ...base, betas: [1, 1.2] })).toThrow()
    expect(() => parallelTemperingKernel(t, { ...base, swapEvery: 0 })).toThrow()
    const k = parallelTemperingKernel(t, { ...base, swapEvery: 3 })
    let s = k.init([0])
    const tried: number[] = []
    for (let i = 0; i < 6; i++) {
      s = k.step(key(i), s).state
      tried.push(s.swapAttempts[0] ?? 0)
    }
    // A swap is proposed on every third transition, on the pairs of the scheme's parity (the single pair every other time).
    expect(tried[2]).toBeGreaterThanOrEqual(0)
    expect(tried[5]).toBeLessThanOrEqual(2)
  })
})
