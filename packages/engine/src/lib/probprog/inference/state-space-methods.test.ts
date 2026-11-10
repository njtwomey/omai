// The state-space filter method against the Kalman filter on a linear-Gaussian model (every proposal and every resampling scheme
// estimates the exact log likelihood and the filtered mean) and on a nonlinear model (the proposals agree on the evidence),
// resuming, snapshots on a fake clock, cancellation, and `check`/`make` saying what the model lacks for a proposal.
import { describe, expect, it } from 'vitest'
import { expectCancelAndContinue, expectResumeEqual, expectSnapshotCadence } from '../../../testing/methods.ts'
import { filtering } from '../../dynamics/index.ts'
import { key, normal } from '../../random/index.ts'
import { add, div, item, mul, square, toFlat, type Value } from '../../tensor/index.ts'
import { session } from '../../trace/index.ts'
import { stateSpaceFilterMethod } from './state-space-methods.ts'

const PHI = 0.9
const Q = 0.09
const R = 0.04
const linear = { f: (x: Value) => mul(x, PHI), h: (x: Value) => x, Q: [[Q]], R: [[R]], x0: [0], P0: [[1]] }
const kalman = { F: [[PHI]], H: [[1]], Q: [[Q]], R: [[R]], x0: [0], P0: [[1]] }
const Z = (() => {
  const e = toFlat(normal(key(3), [10]))
  const o = toFlat(normal(key(4), [10]))
  let x = 0.3
  return Array.from({ length: 10 }, (_, t) => {
    x = PHI * x + Math.sqrt(Q) * (e[t] ?? 0)
    return [x + Math.sqrt(R) * (o[t] ?? 0)]
  })
})()
const exact = filtering.kalmanFilter(kalman, Z)
const model = filtering.gaussianGuidedModel(linear, { observationMatrix: [[1]] })

describe('the state-space filter method on a linear-Gaussian model', () => {
  const method = stateSpaceFilterMethod(model, Z)

  it('estimates the exact log likelihood and the filtered mean with every proposal', () => {
    // The auxiliary filter's first stage uses a point prediction, which is too sharp when the process noise exceeds the
    // observation noise (as here): it stays unbiased but its evidence estimate varies several times more.
    const tolerance = { bootstrap: 0.2, auxiliary: 1, guided: 0.1 }
    for (const proposal of ['bootstrap', 'auxiliary', 'guided'] as const) {
      const s = session(method, { steps: 20, mode: 'eager', seed: 2, config: { particles: 1500, proposal } })
      expect(s.status, proposal).toBe('converged')
      const post = method.posterior?.(s.state, s.config)
      expect(Math.abs((post?.logEvidence?.value ?? 0) - item(exact.logLikelihood)), proposal).toBeLessThan(
        tolerance[proposal],
      )
      const mean = toFlat(post?.mean('x') ?? (undefined as never))[0] ?? 0
      expect(Math.abs(mean - (toFlat(exact.means).at(-1) ?? 0)), proposal).toBeLessThan(0.05)
    }
  })

  it('every resampling scheme and threshold gives a consistent estimate', () => {
    for (const resampling of ['multinomial', 'systematic', 'stratified', 'residual'] as const)
      for (const essThreshold of [0.3, 1]) {
        const s = session(method, {
          steps: 20,
          mode: 'eager',
          seed: 5,
          config: { particles: 800, resampling, essThreshold, proposal: 'guided' },
        })
        expect(
          Math.abs((method.posterior?.(s.state, s.config).logEvidence?.value ?? 0) - item(exact.logLikelihood)),
          resampling,
        ).toBeLessThan(0.2)
      }
  })

  it('logs the effective sample size and the evidence at every step, and reads a partial posterior mid-run', () => {
    const tape = session(method, { steps: 10, mode: 'eager', config: { particles: 100 }, logs: 'all' }).run()
    const names = new Set(tape.events.flatMap((e) => (e.type === 'metric' || e.type === 'tensor' ? [e.name] : [])))
    for (const p of ['pf/ess', 'pf/log-evidence', 'pf/resampled']) expect(names.has(p), p).toBe(true)
    const s = session(method, { steps: 10, config: { particles: 100 } })
    s.advance({ steps: 4 })
    const partial = method.posterior?.(s.state, s.config)
    expect(partial?.status).toBe('stopped')
    expect(partial?.diagnostics['step']).toBe(4)
  })

  it('resumes from a serialised state, snapshots on the cadence of a clock and cancels between steps', () => {
    const config = { particles: 60, proposal: 'guided' as const }
    expectResumeEqual(method, { config, steps: 10, at: 4 })
    expectSnapshotCadence(method, { config, steps: 10, budgetMs: 300, everyMs: 100, tick: 10 })
    expectCancelAndContinue(method, { config, steps: 10, cancelAfterSlices: 3 })
  })

  it('says what the model lacks for a proposal', () => {
    const plain = filtering.gaussianParticleModel(linear)
    const { guide: _g, predict: _p, ...bare } = plain
    expect(stateSpaceFilterMethod(bare, Z).check?.(bare)?.reason).toBe('proposals available: bootstrap')
    expect(method.check?.(model)?.reason).toBe('proposals available: bootstrap, auxiliary, guided')
    expect(() =>
      session(stateSpaceFilterMethod(bare, Z), { steps: 1, mode: 'eager', config: { proposal: 'guided' } }),
    ).toThrow(/guide/)
    expect(() =>
      session(stateSpaceFilterMethod(bare, Z), { steps: 1, mode: 'eager', config: { proposal: 'auxiliary' } }),
    ).toThrow(/predict/)
    expect(stateSpaceFilterMethod(model, []).check?.(model)?.applies).toBe(false)
  })
})

describe('the state-space filter method on a nonlinear model', () => {
  const nonlinear = {
    f: (x: Value) => add(mul(x, 0.5), mul(div(x, add(square(x), 1)), 25)),
    h: (x: Value) => div(square(x), 20),
    Q: [[4]],
    R: [[0.5]],
    x0: [0],
    P0: [[1]],
  }
  const series = (() => {
    const e = toFlat(normal(key(21), [10]))
    const o = toFlat(normal(key(22), [10]))
    let x = 0.1
    return Array.from({ length: 10 }, (_, t) => {
      x = 0.5 * x + (25 * x) / (1 + x * x) + 2 * (e[t] ?? 0)
      return [(x * x) / 20 + Math.sqrt(0.5) * (o[t] ?? 0)]
    })
  })()

  it('the proposals agree on the evidence within the spread of the estimates', () => {
    const m = stateSpaceFilterMethod(filtering.gaussianGuidedModel(nonlinear), series)
    const z = (proposal: string, particles: number, seed: number): number => {
      const s = session(m, { steps: 20, mode: 'eager', seed, config: { particles, proposal } })
      return m.posterior?.(s.state, s.config).logEvidence?.value ?? Number.NaN
    }
    const reference = z('bootstrap', 4000, 1)
    expect(Math.abs(z('guided', 150, 2) - reference)).toBeLessThan(1.5)
    expect(Math.abs(z('auxiliary', 600, 3) - reference)).toBeLessThan(1.5)
  })
})
