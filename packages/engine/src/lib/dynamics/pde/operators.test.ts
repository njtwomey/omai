import { describe, expect, it } from 'vitest'
import { grad } from '../../autodiff/index.ts'
import {
  item,
  mean,
  mul,
  reshape,
  sin,
  sliceAxis,
  stack,
  sub,
  sum,
  tanh,
  tensor,
  toArray,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import {
  curlOf,
  divergenceOf,
  gradientOf,
  laplacianOf,
  partialDerivative,
  pdeResidual,
  timeDerivativeOf,
} from './operators.ts'

const col = (p: Value, i: number): Value => reshape(sliceAxis(p, 1, i, i + 1), [-1])

describe('autodiff differential operators', () => {
  const u = (p: Value): Value => mul(sin(col(p, 0)), mul(col(p, 1), col(p, 1)))
  const pts = tensor([
    [0.3, 1.2],
    [1.1, -0.4],
  ])

  it('match closed forms', () => {
    const g = toArray(gradientOf(u)(pts)) as number[][]
    expect(g[0]?.[0]).toBeCloseTo(Math.cos(0.3) * 1.44, 12)
    expect(g[1]?.[1]).toBeCloseTo(Math.sin(1.1) * 2 * -0.4, 12)
    const lap = toFlat(laplacianOf(u)(pts))
    expect(lap[0]).toBeCloseTo(-Math.sin(0.3) * 1.44 + 2 * Math.sin(0.3), 12)
    expect(toFlat(partialDerivative(u, 1, 3)(pts))).toEqual([0, 0])
    const F = (p: Value): Value => mul(p, tensor([2, 3]))
    expect(toFlat(divergenceOf(F)(pts))).toEqual([5, 5])
    // F = grad(x y z) is curl-free.
    const rot = (p: Value): Value => {
      const [x, y, z] = [col(p, 0), col(p, 1), col(p, 2)]
      return stack([mul(y, z), mul(x, z), mul(x, y)], 1)
    }
    expect(toFlat(curlOf(rot)(tensor([[1, 2, 3]])))).toEqual([0, 0, 0])
    expect(() => curlOf((p) => p)(tensor([[1, 2, 3, 4]]))).toThrow()
    expect(toFlat(timeDerivativeOf((t, x) => mul(mul(t, t), sum(x, 1)))(tensor([2]), tensor([[3]])))).toEqual([12])
  })

  it('a physics-informed residual is differentiable in the network parameters', () => {
    // u_theta(x) = tanh(w x) on [0, 1]; residual of u'' = 0. The loss gradient is checked against differences.
    const loss = (w: Value): Value => {
      const net = (p: Value): Value => tanh(mul(col(p, 0), w))
      const r = pdeResidual(net, (d) => d.laplacian, tensor([[0.2], [0.5], [0.9]]))
      return mean(mul(r, r))
    }
    const g = item(grad(loss)(tensor(1.3)))
    const h = 1e-6
    const fd = (item(loss(tensor(1.3 + h))) - item(loss(tensor(1.3 - h)))) / (2 * h)
    expect(Math.abs(g - fd)).toBeLessThan(1e-6)
    expect(
      toFlat(
        pdeResidual(
          (p) => sum(mul(p, p), 1),
          (d) => sub(d.laplacian, 4),
          tensor([[1, 1]]),
        ),
      ),
    ).toEqual([0])
  })
})
