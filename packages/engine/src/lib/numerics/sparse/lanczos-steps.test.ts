import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { runAlgorithm } from '../../trace/index.ts'
import * as S from './index.ts'

/** A diagonal matrix with a well separated top and a cluster inside, as a CSR matrix. */
const spectrum = [0, 0.3, 0.31, 0.5, 0.52, 0.55, 0.7, 0.9, 1.4, 2, 3, 5]
const a = S.diagCsr(spectrum)
const v0 = spectrum.map((_, i) => 1 + 0.1 * Math.sin(i))

describe('stepped Lanczos', () => {
  it('matches the whole-process lanczos with full reorthogonalisation', () => {
    const whole = S.lanczos(a, 6, { v0 })
    const state = S.lanczosSteps(a, 6, { v0 })
    const [alpha, beta] = [toFlat(whole.alpha), toFlat(whole.beta)]
    state.alpha.forEach((v, i) => {
      expect(v).toBeCloseTo(alpha[i] ?? 0, 10)
    })
    state.beta.forEach((v, i) => {
      expect(v).toBeCloseTo(beta[i] ?? 0, 10)
    })
  })

  it('finds the extreme Ritz values first, with residuals that fall as they converge', () => {
    const early = S.lanczosSteps(a, 5, { v0 })
    const later = S.lanczosSteps(a, 9, { v0 })
    const top = (s: S.LanczosStepsState): number => s.ritzValues[s.ritzValues.length - 1] ?? 0
    expect(Math.abs(top(later) - 5)).toBeLessThan(Math.abs(top(early) - 5))
    expect(Math.abs(top(later) - 5)).toBeLessThan(1e-4)
    // The extreme pairs have converged further than the interior ones.
    const res = later.ritzResiduals
    expect(Math.min(res[0] ?? 1, res[res.length - 1] ?? 1)).toBeLessThan(Math.min(...res.slice(2, -2)) + 1e-12)
  })

  it('finds every eigenvalue when the Krylov space fills the space', () => {
    const state = S.lanczosSteps(a, 40, { v0 })
    expect(state.alpha).toHaveLength(spectrum.length)
    state.ritzValues.forEach((v, i) => {
      expect(v).toBeCloseTo(spectrum[i] ?? 0, 8)
    })
  })

  it('keeps the basis orthonormal with reorthogonalisation and loses it without', () => {
    const full = S.lanczosSteps(a, 11, { v0 })
    const none = S.lanczosSteps(a, 11, { v0, reorthogonalise: 'none' })
    expect(full.orthogonalityLoss).toBeLessThan(1e-12)
    expect(none.orthogonalityLoss).toBeGreaterThan(1e-8)
  })

  it('stops with an invariant subspace when the start has few eigencomponents', () => {
    const two = S.lanczosSteps(S.diagCsr([1, 1, 4, 4, 4]), 10, { v0: [1, 0, 1, 0, 0] })
    expect(two.alpha.length).toBeLessThanOrEqual(2)
    expect(two.invariant).toBe(true)
    expect(two.ritzValues.map((v) => Math.round(v * 1e8) / 1e8)).toEqual([1, 4])
  })

  it('is deterministic in the key and serialises: pause, store and resume', () => {
    const alg = S.lanczosAlgorithm(a)
    const straight = runAlgorithm(alg, { maxSteps: 8, key: key(3) })
    const again = runAlgorithm(alg, { maxSteps: 8, key: key(3) })
    expect(again.state.alpha).toEqual(straight.state.alpha)
    const first = runAlgorithm(alg, { maxSteps: 3, key: key(3) })
    const stored = JSON.parse(JSON.stringify(alg.serialise?.(first.state)))
    const resumed = runAlgorithm(alg, { maxSteps: 5, key: key(3), start: { state: stored, step: 3, serialised: true } })
    expect(resumed.state.alpha).toEqual(straight.state.alpha)
    expect(resumed.state.ritzValues).toEqual(straight.state.ritzValues)
  })
})
