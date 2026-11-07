import { describe, expect, it } from 'vitest'
import { grad } from '../autodiff/index.ts'
import { eigvalsh } from '../numerics/linalg/index.ts'
import { key, normal } from '../random/index.ts'
import {
  matmul,
  mean,
  mul,
  norm,
  ravel,
  relu,
  square,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  tensor,
  toFlat,
  transpose,
  type Value,
} from '../tensor/index.ts'
import * as O from './index.ts'

const flat = (v: unknown): number[] => toFlat(v as Value)

/** A bias-free two-layer ReLU network's mean squared error: positively homogeneous in each layer. */
const x = normal(key(1), [20, 3]) as Tensor
const targets = normal(key(2), [20, 2]) as Tensor
const netLoss = (p: unknown): Value => {
  const { w1, w2 } = p as { w1: Value; w2: Value }
  return mean(square(sub(matmul(relu(matmul(x, transpose(w1))), transpose(w2)), targets)))
}
const params = { w1: normal(key(3), [5, 3]) as Tensor, w2: normal(key(4), [2, 5]) as Tensor }

describe('slices', () => {
  it('equal the loss at the point where alpha is zero, in one and two dimensions', () => {
    const at = flat(netLoss(params))[0] as number
    const d1 = O.randomDirection(params, key(5))
    const d2 = O.randomDirection(params, key(6))
    expect(flat(O.slice1d(netLoss, params, d1, [0, 0.3]).losses)[0]).toBe(at)
    const plane = O.slice2d(netLoss, params, d1, d2, [-0.5, 0, 0.5], [0, 0.25])
    expect(plane.losses.shape).toEqual([3, 2])
    expect(flat(plane.losses)[2]).toBe(at)
    expect(plane.losses.data[1 * 2 + 0]).toBe(at)
  })

  it('a 2-D slice of a quadratic matches the closed form', () => {
    const q = (v: unknown): Value => sum(mul(square(v as Value), tensor([1, 4])))
    const s = O.slice2d(q, tensor([1, 1]), tensor([1, 0]), tensor([0, 1]), [0, 1], [0, 1, 2])
    expect(flat(s.losses)).toEqual([1 + 4, 1 + 16, 1 + 36, 4 + 4, 4 + 16, 4 + 36])
  })

  it('filter normalisation gives every filter of the direction the norm of the matching filter, and zeroes biases', () => {
    const p = { w: normal(key(1), [4, 6]) as Tensor, b: normal(key(2), [4]) as Tensor }
    const d = O.randomDirection(p, key(3)) as { w: Tensor; b: Tensor }
    for (let r = 0; r < 4; r++) {
      const dn = Math.hypot(...flat(d.w).slice(6 * r, 6 * r + 6))
      const pn = Math.hypot(...flat(p.w).slice(6 * r, 6 * r + 6))
      expect(dn).toBeCloseTo(pn, 12)
    }
    expect(flat(d.b).every((v) => v === 0)).toBe(true)
    const kept = O.randomDirection(p, key(3), { biases: 'normalise' }) as { b: Tensor }
    expect(flat(norm(kept.b))[0]).toBeCloseTo(flat(norm(p.b))[0] as number, 12)
    const layer = O.randomDirection(p, key(3), { mode: 'layer' }) as { w: Tensor }
    expect(flat(norm(layer.w))[0]).toBeCloseTo(flat(norm(p.w))[0] as number, 12)
  })

  it('filter normalisation along another axis normalises columns', () => {
    const p = tensor([
      [3, 0],
      [4, 5],
    ])
    const d = O.filterNormalise(
      tensor([
        [1, 1],
        [1, 1],
      ]),
      p,
      { filterAxis: 1 },
    )
    const f = flat(d)
    expect(Math.hypot(f[0] as number, f[2] as number)).toBeCloseTo(5, 12)
    expect(Math.hypot(f[1] as number, f[3] as number)).toBeCloseTo(5, 12)
  })

  it('filter-normalised slices do not depend on the scale of the weights; raw ones do', () => {
    // Scale layer one up and layer two down: the network, and so its loss at the point, is unchanged.
    const c = 7
    const scaled = { w1: mul(params.w1, c) as Tensor, w2: mul(params.w2, 1 / c) as Tensor }
    expect(flat(netLoss(scaled))[0]).toBeCloseTo(flat(netLoss(params))[0] as number, 12)
    const alphas = [-1, -0.5, 0, 0.5, 1]
    const a = O.randomSlice(netLoss, params, key(9), { alphas }).slice as O.Slice1d
    const b = O.randomSlice(netLoss, scaled, key(9), { alphas }).slice as O.Slice1d
    flat(a.losses).forEach((l, i) => {
      expect(flat(b.losses)[i]).toBeCloseTo(l, 9)
    })
    const rawA = O.randomSlice(netLoss, params, key(9), { alphas }, { mode: 'none' }).slice as O.Slice1d
    const rawB = O.randomSlice(netLoss, scaled, key(9), { alphas }, { mode: 'none' }).slice as O.Slice1d
    expect(Math.abs((flat(rawA.losses)[0] as number) - (flat(rawB.losses)[0] as number))).toBeGreaterThan(1e-3)
  })
})

describe('paths between minima', () => {
  // L = (x^2 + y^2 - 1)^2: a ring of minima on the unit circle.
  const ring = (p: unknown): Value => {
    const v = p as Value
    return square(sub(sum(square(v)), 1))
  }
  const a = tensor([-1, 0])
  const b = tensor([1, 0])
  const ts = Array.from({ length: 41 }, (_, i) => i / 40)

  it('the straight line crosses the ring’s hole and a bent path avoids it', () => {
    const line = O.interpolationSlice(ring, a, b, ts)
    expect(flat(line.losses)[0]).toBeCloseTo(0, 14)
    expect(flat(line.losses)[20]).toBeCloseTo(1, 12)
    expect(O.barrier(line)).toBeCloseTo(1, 12)
    const bent = O.bezierSlice(ring, a, tensor([0, 2]), b, ts)
    expect(O.barrier(bent)).toBeLessThan(0.1)
    expect(O.barrier({ alphas: [0, 0.5, 1], losses: [1, 3, 1] })).toBe(2)
    expect(O.barrier({ alphas: [0, 1], losses: [1, 3] })).toBe(0)
  })
})

describe('sharpness', () => {
  // The top Hessian eigenvalue of the network loss, against the dense Hessian.
  const flatLoss = O.flatObjective(netLoss, params)

  it('Lanczos and power iteration match the dense Hessian’s largest eigenvalue', () => {
    const h = flatLoss.hessian(flatLoss.x0)
    const dense = flat(
      eigvalsh(
        tensor(Array.from({ length: flatLoss.n }, (_, i) => Array.from(h.slice(i * flatLoss.n, (i + 1) * flatLoss.n)))),
      ),
    )
    const top = Math.max(...dense)
    const lanczos = O.sharpness(netLoss, params)
    expect(lanczos.value).toBeCloseTo(top, 7)
    const eigenvector = Array.from(ravel(lanczos.vector).flat)
    // H v = lambda v for the returned eigenvector.
    const hv = flatLoss.hvp(flatLoss.x0, eigenvector)
    hv.forEach((q, i) => {
      expect(q).toBeCloseTo(top * (eigenvector[i] as number), 5)
    })
    if (Math.abs(top) >= Math.abs(Math.min(...dense))) {
      expect(O.sharpness(netLoss, params, { method: 'power', iterations: 5000, tol: 1e-13 }).value).toBeCloseTo(top, 6)
    }
  })

  it('a quadratic’s sharpness is its largest curvature, and gradient descent is stable below 2 / sharpness only', () => {
    const q = (v: unknown): Value => sum(mul(square(v as Value), tensor([1, 3])))
    const s = O.sharpness(q, tensor([0.5, 0.5])).value
    expect(s).toBeCloseTo(6, 8)
    const run = (eta: number): number => {
      let v = [1, 1]
      for (let i = 0; i < 60; i++) v = [(v[0] as number) * (1 - eta * 2), (v[1] as number) * (1 - eta * 6)]
      return Math.hypot(...v)
    }
    expect(O.stabilityThreshold(0.3)).toBeCloseTo(2 / 0.3, 14)
    expect(run(0.9 * (2 / s))).toBeLessThan(1e-3)
    expect(run(1.05 * (2 / s))).toBeGreaterThan(1)
    expect(O.sharpness(q, tensor([0.5, 0.5]), { method: 'power' }).value).toBeCloseTo(6, 8)
  })

  it('a single parameter’s sharpness is its second derivative', () => {
    expect(O.sharpness((v) => mul(square(v as Value), 2.5), tensor([3])).value).toBeCloseTo(5, 10)
  })

  it('the grids and the start vector take tensor-likes (migration, D-009): tensor, typed array and list agree, rank 2 is a shape error, a traced one is not differentiable', () => {
    const q = (v: unknown): Value => sum(mul(square(v as Value), tensor([1, 4])))
    const p = tensor([1, 1])
    const d = tensor([1, 0])
    const rules: [string, (g: TensorLike) => unknown][] = [
      ['slice1d', (g) => O.slice1d(q, p, d, g)],
      ['slice2d', (g) => O.slice2d(q, p, d, tensor([0, 1]), g, g).losses],
      ['randomSlice', (g) => O.randomSlice(q, p, key(0), { alphas: g, betas: g }).slice],
      ['interpolationSlice', (g) => O.interpolationSlice(q, p, tensor([2, 2]), g)],
      ['bezierSlice', (g) => O.bezierSlice(q, p, tensor([0, 0]), tensor([2, 2]), g)],
      ['sharpness', (g) => O.sharpness(q, p, { method: 'power', v0: g }).value],
    ]
    const grid = [0, 0.5, 1]
    for (const [name, f] of rules) {
      const g = name === 'sharpness' ? [1, 0.5] : grid
      const expected = JSON.stringify(f(g))
      expect(JSON.stringify(f(Float64Array.from(g))), name).toBe(expected)
      expect(JSON.stringify(f(tensor(g))), name).toBe(expected)
      expect(() => f(tensor([g, g])), name).toThrow(expect.objectContaining({ code: 'shape' }))
      expect(
        () =>
          grad((a: Value) => {
            f(a as TensorLike)
            return sum(a)
          })(tensor(g)),
        name,
      ).toThrow(expect.objectContaining({ code: 'not-differentiable' }))
    }
  })
})
