import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key, uniform } from '../random/index.ts'
import { toFlat, type Value } from '../tensor/index.ts'
import * as NN from './index.ts'

const flat = (v: unknown): number[] => toFlat(v as Value)

describe('interval bound propagation is sound', () => {
  it('contains the network output at every sampled point of the box', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 0, max: 500 }),
        fc.double({ min: 0, max: 1.5, noNaN: true }),
        fc.constantFrom('relu', 'tanh', 'sigmoid'),
        (seed, radius, activation) => {
          const act = (): NN.Module =>
            activation === 'relu' ? new NN.ReLU() : activation === 'tanh' ? new NN.Tanh() : new NN.Sigmoid()
          const net = NN.mlp([3, 6, 6, 2], { key: key(seed), activation: act })
          const centre = flat(uniform(key(seed + 1), [3], { low: -1, high: 1 }))
          const lo = [centre.map((c) => c - radius)]
          const hi = [centre.map((c) => c + radius)]
          const bounds = NN.intervalBoundPropagation(net, lo, hi).output
          const l = flat(bounds.lower)
          const u = flat(bounds.upper)
          const draws = flat(uniform(key(seed + 2), [40, 3], { low: -1, high: 1 }))
          for (let s = 0; s < 40; s++) {
            const point = [0, 1, 2].map((k) => (centre[k] as number) + radius * (draws[3 * s + k] as number))
            const y = flat(net.forward([point]))
            y.forEach((v, o) => {
              expect(v).toBeGreaterThanOrEqual((l[o] as number) - 1e-9)
              expect(v).toBeLessThanOrEqual((u[o] as number) + 1e-9)
            })
          }
        },
      ),
      { numRuns: 60 },
    )
  })

  it('a certificate is never wrong: certified points keep their label at sampled perturbations', () => {
    fc.assert(
      fc.property(fc.integer({ min: 0, max: 300 }), fc.double({ min: 0, max: 0.4, noNaN: true }), (seed, eps) => {
        const net = NN.mlp([2, 5, 3], { key: key(seed) })
        const x = flat(uniform(key(seed + 1), [1, 2], { low: -1, high: 1 }))
        const label = flat(NN.mlp([2, 5, 3], { key: key(seed) }).forward([x])).reduce(
          (b, v, i, a) => (v > (a[b] as number) ? i : b),
          0,
        )
        const cert = NN.ibpCertify(net, [x], [label], eps)
        if (!cert.certified[0]) return
        const draws = flat(uniform(key(seed + 2), [50, 2], { low: -eps, high: eps }))
        for (let s = 0; s < 50; s++) {
          const y = flat(
            net.forward([
              [(x[0] as number) + (draws[2 * s] as number), (x[1] as number) + (draws[2 * s + 1] as number)],
            ]),
          )
          expect(y.reduce((b, v, i, a) => (v > (a[b] as number) ? i : b), 0)).toBe(label)
        }
      }),
      { numRuns: 80 },
    )
  })
})
