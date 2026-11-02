import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../../testing/check.ts'
import { jacobian, jvp } from '../../autodiff/index.ts'
import { norm, slogdet } from '../../numerics/linalg/index.ts'
import { key, normal } from '../../random/index.ts'
import {
  add,
  eye,
  item,
  matmul,
  mul,
  sliceAxis,
  squeeze,
  tanh,
  tensor,
  toArray,
  toFlat,
  type Value,
  zeros,
} from '../../tensor/index.ts'
import * as B from './index.ts'

const flatten = (v: Value): number[] => Array.from(toFlat(v))
const closeAll = (a: readonly number[], b: readonly number[], digits: number, label: string): void => {
  expect(a.length, label).toBe(b.length)
  for (const [i, v] of a.entries()) expect(v, `${label}[${i}]`).toBeCloseTo(b[i] ?? Number.NaN, digits)
}

/**
 * The checks every vector bijector must pass at a batch of inputs: the round trip in both directions, the forward
 * log-determinant equal to log |det J| of the autodiff Jacobian of each row, and the inverse log-determinant equal to its
 * negative at the preimage.
 */
function checkVectorBijector(name: string, b: B.Bijector, x: Value, digits = 9): void {
  const y = b.forward(x)
  closeAll(flatten(b.inverse(y)), flatten(x), digits, `${name} inverse(forward)`)
  closeAll(flatten(b.forward(b.inverse(y))), flatten(y), digits, `${name} forward(inverse)`)
  const rows = toArray(x) as number[][]
  const ldj = flatten(b.forwardLogDetJacobian(x))
  for (const [i, row] of rows.entries()) {
    const j = jacobian((v: Value) => b.forward(v))(tensor(row))
    closeAll([ldj[i] ?? Number.NaN], [item(slogdet(j).logabsdet)], digits, `${name} log det row ${i}`)
  }
  closeAll(
    flatten(b.inverseLogDetJacobian(y)),
    ldj.map((v) => -v),
    digits,
    `${name} inverse log det`,
  )
}

const x3 = tensor([
  [0.4, -1.1, 0.7],
  [-0.3, 0.2, 1.4],
  [1.2, 0.9, -0.6],
])

describe('planar, radial and normalisation layers', () => {
  it('planar flows invert and have the log-determinant of their Jacobian', () => {
    checkVectorBijector('planar', B.planar([0.8, -0.5, 0.3], [1.1, 0.4, -0.7], 0.2), x3)
    // A strongly contracting direction: w . u_hat close to -1.
    checkVectorBijector('planar contracting', B.planar([-4, 0, 0], [1, 0.2, 0], -0.3), x3)
    // An expanding direction with w . u_hat above 1, where the root of the inverse lies further than 1 from w . y
    // (a counterexample that fast-check found).
    checkVectorBijector(
      'planar expanding',
      B.planar([0, -1.0370896934973455, 0], [0, -1.9999999971987505, 0], 0),
      tensor([[0, -0.6097326246182959, 0]]),
    )
  })

  it('radial flows invert and have the log-determinant of their Jacobian', () => {
    checkVectorBijector('radial', B.radial([0.1, -0.2, 0.3], 0.8, 0.5), x3)
    checkVectorBijector('radial contracting', B.radial([0, 0, 0], 1.5, -3), x3)
  })

  it('has gradients for the inverse and log-determinant in the parameters', () => {
    expectConformant((u, w) => B.planar(u, w, 0.2).inverse(x3), [tensor([0.8, -0.5, 0.3]), tensor([1.1, 0.4, -0.7])])
    expectConformant((u) => B.planar(u, [1.1, 0.4, -0.7], 0.2).forwardLogDetJacobian(x3), [tensor([0.8, -0.5, 0.3])])
    expectConformant((a, b) => B.radial([0.1, -0.2, 0.3], a, b).inverse(x3), [tensor(0.8), tensor(0.5)])
  })

  it('normalises a batch with ActNorm initialisation and keeps the log-determinant', () => {
    const data = tensor([
      [1, 10],
      [3, 14],
      [5, 18],
      [7, 22],
    ])
    const { bias, logScale } = B.actNormInitialise(data)
    const layer = B.actNorm(bias, logScale)
    const out = toArray(layer.forward(data)) as number[][]
    for (const c of [0, 1]) {
      const column = out.map((r) => r[c] ?? 0)
      const m = column.reduce((a, v) => a + v, 0) / 4
      expect(m).toBeCloseTo(0, 6)
      expect(Math.sqrt(column.reduce((a, v) => a + (v - m) ** 2, 0) / 4)).toBeCloseTo(1, 4)
    }
    checkVectorBijector('actNorm', layer, data)
    expect(item(B.actNorm([0, 0], [0.5, 0.25], { positions: 10 }).forwardLogDetJacobian([1, 1]))).toBeCloseTo(7.5, 12)
  })
})

describe('Glow linear layers', () => {
  const w = tensor([
    [0.9, 0.3, -0.2],
    [0.4, 1.2, 0.5],
    [-0.1, 0.6, 0.8],
  ])

  it('applies an invertible matrix with log |det W|', () => {
    checkVectorBijector('invertibleLinear', B.invertibleLinear(w), x3)
    expect(toArray(B.invertibleLinear(w).forward([1, 0, 0]))).toEqual([0.9, 0.4, -0.1])
  })

  it('reconstructs the matrix from its LU parameters, with a cheap log-determinant, for a matrix needing a pivot', () => {
    const pivoting = tensor([
      [0, 1.5, 0.2],
      [2, 0.1, -0.4],
      [0.3, -0.7, 1],
    ])
    for (const m of [w, pivoting]) {
      const layer = B.luLinear(B.luParameters(m))
      const columns = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ].map((e) => flatten(layer.forward(e)))
      const original = toArray(m) as number[][]
      for (const [c, col] of columns.entries())
        closeAll(
          col,
          original.map((row) => row[c] ?? 0),
          12,
          `column ${c}`,
        )
      closeAll([item(layer.forwardLogDetJacobian([1, 1, 1]))], [item(slogdet(m).logabsdet)], 12, 'log det')
      checkVectorBijector('luLinear', layer, x3, 10)
    }
  })

  it('shares the matrix over positions', () => {
    expect(item(B.invertibleLinear(w, { positions: 16 }).forwardLogDetJacobian([1, 1, 1]))).toBeCloseTo(
      16 * item(slogdet(w).logabsdet),
      12,
    )
  })
})

describe('autoregressive flows with a MADE conditioner', () => {
  const params = B.madeParameters(3, [12, 12], key(4))
  // Larger output weights so the flow is far from the identity.
  const strong: B.MadeParameters = {
    weights: params.weights.map((w, l) => (l === params.weights.length - 1 ? mul(w, 40) : w)),
    biases: params.biases,
  }

  it('has a conditioner whose output i depends only on inputs before i', () => {
    const conditioner = B.madeConditioner(strong)
    const jacobianShift = toArray(jacobian((v: Value) => conditioner(v).shift)(tensor([0.3, -0.5, 0.9]))) as number[][]
    const jacobianScale = toArray(
      jacobian((v: Value) => conditioner(v).logScale)(tensor([0.3, -0.5, 0.9])),
    ) as number[][]
    for (const j of [jacobianShift, jacobianScale])
      for (const [i, row] of j.entries())
        for (const [c, v] of row.entries()) if (c >= i) expect(v, `d out_${i} / d in_${c}`).toBe(0)
    // And it does depend on earlier inputs.
    expect(Math.abs(jacobianShift[2]?.[0] ?? 0) + Math.abs(jacobianShift[2]?.[1] ?? 0)).toBeGreaterThan(1e-3)
  })

  it('makes MAF and IAF layers that invert and have the triangular Jacobian log-determinant', () => {
    checkVectorBijector('maf', B.autoregressive(B.madeConditioner(strong)), x3)
    checkVectorBijector('iaf', B.inverseAutoregressive(B.madeConditioner(strong)), x3)
    expect(B.madeMasks(3, [12, 12]).length).toBe(3)
  })

  it('is differentiable in the network parameters', () => {
    expectConformant(
      (w0) =>
        B.autoregressive(
          B.madeConditioner({ weights: [w0, ...strong.weights.slice(1)], biases: strong.biases }),
        ).forwardLogDetJacobian(x3),
      [strong.weights[0] as Value],
    )
  })
})

describe('rational-quadratic splines', () => {
  const bins = 6
  const raw = normal(key(9), [3 * (3 * bins - 1)])
  const parameters = B.splineParametersFromVector(raw, bins)
  const spline = B.rationalQuadraticSpline(parameters)
  const inputs = tensor([
    [-2.7, -0.4, 0.05],
    [0.3, 2.9, -3.0],
    [1.5, -1.2, 2.2],
  ])

  it('is monotone, the identity outside the bound, and continuous with unit slope at the bound', () => {
    const grid = Array.from({ length: 121 }, (_, i) => -6 + i * 0.1)
    const values = grid.map((u) => item(sliceAxis(spline.forward(tensor([u, u, u])), 0, 0, 1)))
    for (let i = 1; i < values.length; i++) expect(values[i] ?? 0).toBeGreaterThan(values[i - 1] ?? 0)
    for (const [i, u] of grid.entries()) if (Math.abs(u) > 3.0001) expect(values[i]).toBeCloseTo(u, 12)
    expect(item(sliceAxis(spline.forward(tensor([3, 3, 3])), 0, 0, 1))).toBeCloseTo(3, 10)
    expect(item(sliceAxis(spline.forward(tensor([-3, -3, -3])), 0, 0, 1))).toBeCloseTo(-3, 10)
  })

  it('inverts exactly and has the derivative of the forward map as its log-determinant (per coordinate)', () => {
    const y = spline.forward(inputs)
    closeAll(flatten(spline.inverse(y)), flatten(inputs), 9, 'inverse(forward)')
    const derivative = flatten(
      jvp(
        (v: Value) => spline.forward(v),
        inputs,
        tensor([
          [1, 1, 1],
          [1, 1, 1],
          [1, 1, 1],
        ]),
      ).tangent,
    )
    closeAll(
      flatten(spline.forwardLogDetJacobian(inputs)),
      derivative.map((d) => Math.log(d)),
      8,
      'log derivative',
    )
    // Outside the bound the map is the identity with zero log-derivative.
    closeAll(flatten(spline.forwardLogDetJacobian(tensor([[5, -5, 4]]))), [0, 0, 0], 14, 'tails')
  })

  it('makes a coupling layer that inverts and has the log-determinant of its Jacobian', () => {
    const conditioner = (xa: Value): B.SplineParameters => {
      // A fixed little network of xa: 1 -> (3 bins' worth for 2 coordinates).
      const w = tensor([Array.from({ length: 2 * (3 * 4 - 1) }, (_, i) => 0.3 * Math.sin(i + 1))])
      const bias = tensor(Array.from({ length: 2 * (3 * 4 - 1) }, (_, i) => 0.2 * Math.cos(2 * i)))
      return B.splineParametersFromVector(add(matmul(xa, w), bias), 4)
    }
    const layer = B.rationalQuadraticCoupling(1, conditioner)
    checkVectorBijector('spline coupling', layer, x3, 9)
    checkVectorBijector(
      'spline coupling with a tail',
      layer,
      tensor([
        [0.2, 4.5, -0.6],
        [-1.1, 1.7, 3.4],
      ]),
      9,
    )
  })

  it('has gradients of the forward map and the log-derivative in the spline parameters', () => {
    expectConformant(
      (u) =>
        B.rationalQuadraticSplineMap(
          inputs,
          { widths: zeros([3, 3, bins]), heights: u, derivatives: zeros([3, 3, bins - 1]) },
          false,
        ).value,
      [normal(key(1), [3, 3, bins])],
    )
    expectConformant(
      (u) =>
        B.rationalQuadraticSplineMap(
          inputs,
          { widths: u, heights: zeros([3, 3, bins]), derivatives: zeros([3, 3, bins - 1]) },
          true,
        ).logDerivative,
      [normal(key(2), [3, 3, bins])],
    )
  })
})

describe('residual flows', () => {
  const w = B.lipschitzNormalise(
    tensor([
      [1.4, -0.9, 0.5],
      [0.3, 1.1, -0.6],
      [-0.8, 0.2, 1.3],
    ]),
    0.7,
  )
  const g = (v: Value): Value => tanh(matmul(v, w))

  it('normalises the spectral norm of a weight', () => {
    expect(item(norm(w, 2))).toBeCloseTo(0.7, 12)
    expect(
      item(
        norm(
          B.lipschitzNormalise(
            [
              [0.2, 0],
              [0, 0.1],
            ],
            0.7,
          ),
          2,
        ),
      ),
    ).toBeCloseTo(0.2, 12)
  })

  it('inverts by fixed-point iteration and has the log-determinant of I + J', () => {
    const layer = B.residualFlow(g, { terms: 70 })
    checkVectorBijector('residual flow', layer, x3, 8)
  })

  it('has the power-series log-determinant converge at the Lipschitz rate in the number of terms', () => {
    const x = tensor([0.4, -1.1, 0.7])
    const exact = item(slogdet(add(eye(3), jacobian((v: Value) => g(v))(x))).logabsdet)
    const errors = [5, 10, 20, 40].map((terms) =>
      Math.abs(item(B.residualFlow(g, { terms }).forwardLogDetJacobian(x)) - exact),
    )
    for (let i = 1; i < errors.length; i++) expect(errors[i] ?? 1).toBeLessThan((errors[i - 1] ?? 0) * 0.5)
    expect(errors[3]).toBeLessThan(1e-9)
  })

  it('has an unbiased Hutchinson estimate of the truncated series', () => {
    const x = tensor([[0.4, -1.1, 0.7]])
    const truncated = B.residualFlow(g, { terms: 12 }).forwardLogDetJacobian(x)
    const estimate = B.residualFlow(g, {
      terms: 12,
      estimator: 'hutchinson',
      key: key(3),
      samples: 600,
    }).forwardLogDetJacobian(x)
    expect(Math.abs(item(squeeze(estimate)) - item(squeeze(truncated)))).toBeLessThan(0.06)
    expect(() => B.residualFlow(g, { estimator: 'hutchinson' })).toThrow()
  })

  it('is differentiable in the weights through the exact series', () => {
    expectConformant(
      (m) => B.residualFlow((v) => tanh(matmul(v, mul(m, 0.3))), { terms: 8 }).forwardLogDetJacobian(x3),
      [
        tensor([
          [1.4, -0.9, 0.5],
          [0.3, 1.1, -0.6],
          [-0.8, 0.2, 1.3],
        ]),
      ],
    )
  })
})

describe('sinh-arcsinh and Kumaraswamy bijectors', () => {
  const x = tensor([-2.1, -0.4, 0.3, 1.7])
  it('inverts and has the log-derivative of the forward map', () => {
    for (const [name, b, at] of [
      ['sinhArcsinh', B.sinhArcsinh(0.4, 1.6), x],
      ['sinhArcsinh heavy', B.sinhArcsinh(-0.7, 0.6), x],
      ['kumaraswamy', B.kumaraswamyCdf(2.5, 0.8), tensor([0.05, 0.3, 0.62, 0.97])],
    ] as const) {
      closeAll(flatten(b.inverse(b.forward(at))), flatten(at), 10, `${name} round trip`)
      const derivative = flatten(jvp((v: Value) => b.forward(v), at, tensor([1, 1, 1, 1])).tangent)
      closeAll(
        flatten(b.forwardLogDetJacobian(at)),
        derivative.map((d) => Math.log(d)),
        10,
        `${name} log det`,
      )
    }
    expect(item(B.sinhArcsinh(0, 1).forward(0.7))).toBeCloseTo(0.7, 14)
  })

  it('has the Kumaraswamy CDF equal to the closed form and push a uniform to the Kumaraswamy law', () => {
    const [a, b] = [2, 3]
    const cdf = (u: number): number => 1 - (1 - u ** a) ** b
    closeAll(flatten(B.kumaraswamyCdf(a, b).forward([0.2, 0.5, 0.9])), [0.2, 0.5, 0.9].map(cdf), 14, 'cdf')
    // The Beta(a=1, b=1) special case is the identity.
    closeAll(flatten(B.kumaraswamyCdf(1, 1).forward([0.2, 0.5])), [0.2, 0.5], 14, 'identity')
    expect(item(B.kumaraswamyCdf(a, b).forward(B.kumaraswamyCdf(a, b).inverse(0.3)))).toBeCloseTo(0.3, 12)
  })
})
