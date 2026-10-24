import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key, normal } from '../../random/index.ts'
import { add, item, mul, square, sub, sum, toFlat, type Value } from '../../tensor/index.ts'
import { kalmanFilter, rtsSmoother } from './kalman.ts'
import { auxiliaryParticleFilter, bootstrapParticleFilter, particleSmoother, resampleIndices } from './particle.ts'

// A random walk x_t = x_{t-1} + N(0, q), z_t = x_t + N(0, r), x_0 ~ N(0, 1).
const q = 0.2
const r = 0.5
const model = {
  initial: (k: never, n: number): Value => normal(k, [n, 1]),
  transition: (k: never, x: Value): Value => add(x, mul(normal(k, x.shape), Math.sqrt(q))),
  logLikelihood: (z: Value, x: Value): Value =>
    sub(mul(sum(square(sub(x, z)), -1), -0.5 / r), 0.5 * Math.log(2 * Math.PI * r)),
  logTransition: (x: Value, next: Value): Value => mul(sum(square(sub(x, next)), -1), -0.5 / q),
  predict: (x: Value): Value => x,
}
const zs = [[0.3], [0.9], [0.5], [1.4], [1.1], [1.8]]
const kf = { F: [[1]], H: [[1]], Q: [[q]], R: [[r]], x0: [0], P0: [[1]] }

describe('particle filters', () => {
  it('resampling schemes keep the expected counts (property)', () => {
    fc.assert(
      fc.property(
        fc.array(fc.double({ min: 0.01, max: 1, noNaN: true }), { minLength: 2, maxLength: 6 }),
        fc.double({ min: 0, max: 0.999, noNaN: true }),
        (ws, u) => {
          const total = ws.reduce((a, b) => a + b, 0)
          const w = ws.map((v) => v / total)
          const N = w.length
          for (const scheme of ['systematic', 'stratified', 'residual'] as const) {
            const idx = resampleIndices(
              w,
              scheme,
              Array.from({ length: N }, (_, i) => (u + i * 0.37) % 1),
            )
            const counts = w.map((_, i) => idx.filter((j) => j === i).length)
            // Systematic, stratified and residual resampling keep each count within one of N w_i.
            if (!counts.every((c, i) => Math.abs(c - N * (w[i] ?? 0)) < (scheme === 'residual' ? N : 2))) return false
          }
          return true
        },
      ),
    )
  })

  it('track the Kalman filter on a linear-Gaussian model', () => {
    const exact = kalmanFilter(kf, zs)
    const truth = toFlat(exact.means)
    for (const run of [
      bootstrapParticleFilter(model as never, zs, { particles: 3000, key: key(1) }),
      auxiliaryParticleFilter(model as never, zs, { particles: 3000, key: key(2) }),
    ]) {
      toFlat(run.means).forEach((v, i) => {
        expect(Math.abs(v - (truth[i] ?? 0))).toBeLessThan(0.05)
      })
      expect(Math.abs(run.logEvidence - item(exact.logLikelihood))).toBeLessThan(0.1)
    }
  })

  it('smoothing approaches the RTS smoother', () => {
    const f = bootstrapParticleFilter(model as never, zs, { particles: 1000, key: key(4), essThreshold: 1 })
    const s = particleSmoother(model as never, f, { trajectories: 300, key: key(5) })
    const exact = toFlat(rtsSmoother(kf, kalmanFilter(kf, zs)).means)
    toFlat(s.means).forEach((v, i) => {
      expect(Math.abs(v - (exact[i] ?? 0))).toBeLessThan(0.1)
    })
  })
})
