import { describe, expect, it } from 'vitest'
import { expectRows } from '../../../testing/close.ts'
import type { NestedArray, Value } from '../../tensor/index.ts'
import * as T from '../../tensor/index.ts'
import * as D from '../index.ts'

describe('heavy-tailed distributions against scipy', () => {
  // [name, value under test, reference]: scipy.stats cauchy and halfcauchy; the sinh-arcsinh moments, entropy and mode by
  // quadrature and optimisation in scipy.integrate and scipy.optimize
  const rows: [string, () => Value, NestedArray][] = [
    [
      'cauchy(1, 2): mean and variance do not exist',
      () => T.stack([D.cauchy(1, 2).mean(), D.cauchy(1, 2).variance()]),
      [NaN, NaN],
    ],
    [
      'cauchy(1, 2): mode, entropy',
      () => T.stack([D.cauchy(1, 2).mode(), D.cauchy(1, 2).entropy()]),
      [1, 3.224171427529236],
    ],
    [
      'halfCauchy(2): mean and variance are infinite, mode 0, entropy',
      () =>
        T.stack([
          D.halfCauchy(2).mean(),
          D.halfCauchy(2).variance(),
          D.halfCauchy(2).mode(),
          D.halfCauchy(2).entropy(),
        ]),
      [Infinity, Infinity, 0, 2.5310242469692907],
    ],
    [
      'halfCauchy(2): cdf, sf, ppf, isf',
      () =>
        T.stack([
          D.halfCauchy(2).cdf(1.5),
          D.halfCauchy(2).sf(1.5),
          D.halfCauchy(2).ppf(0.3),
          D.halfCauchy(2).isf(0.3),
        ]),
      [0.4096655293982669, 0.590334470601733, 1.0190508989888576, 3.925221011010301],
    ],
    [
      'sinhArcsinh: mean, variance, entropy by quadrature over the normal',
      () =>
        T.stack([
          D.sinhArcsinh(0.5, 1.3, 0.4, 0.8).mean(),
          D.sinhArcsinh(0.5, 1.3, 0.4, 0.8).variance(),
          D.sinhArcsinh(0.5, 1.3, 0.4, 0.8).entropy(),
        ]),
      [1.015301908832182, 1.0241633504355303, 1.4088186986337439],
    ],
    [
      'sinhArcsinh: log density, cdf and ppf at points',
      () =>
        T.concat([
          D.sinhArcsinh(0.5, 1.3, 0.4, 0.8).logProb(T.tensor([-1, 0.5, 3])),
          D.sinhArcsinh(0.5, 1.3, 0.4, 0.8).cdf(T.tensor([-1, 0.5, 3])),
          D.sinhArcsinh(0.5, 1.3, 0.4, 0.8).ppf(T.tensor([0.1, 0.5, 0.9])),
        ]),
      [
        -3.4324492091590035, -0.9645644975463328, -2.6594035994294782, 0.006929534804738743, 0.3406270770425538,
        0.965740039530045, -0.22725201822652907, 0.923136172720473, 2.4011156651525836,
      ],
    ],
    ['sinhArcsinh: the mode maximises the density', () => D.sinhArcsinh(0.5, 1.3, 0.4, 0.8).mode(), 0.5634464391016897],
  ]
  it('have the reference moments (or NaN and Infinity where they do not exist), entropies, modes and quantiles', () => {
    expectRows(rows, 1e-7)
  })

  // -1 / tan(pi q) evaluated exactly (pi q has a relative error of 1e-16); the old tan(pi (q - 0.5)) lost the tail:
  // -3.183097e9 at 1e-10, -1.63e16 at 1e-17.
  const tails: [string, () => Value, NestedArray][] = [
    ['cauchy ppf(1e-10)', () => D.cauchy(0, 1).ppf(1e-10), -3183098861.8379064],
    ['cauchy ppf(1e-14)', () => D.cauchy(0, 1).ppf(1e-14), -31830988618379.066],
    ['cauchy ppf(1e-17)', () => D.cauchy(0, 1).ppf(1e-17), -3.1830988618379064e16],
    ['cauchy isf(1e-14) is the mirror image', () => D.cauchy(0, 1).isf(1e-14), 31830988618379.066],
    ['cauchy ppf in the middle: 0.25, 0.5, 0.75', () => D.cauchy(0, 1).ppf(T.tensor([0.25, 0.5, 0.75])), [-1, 0, 1]],
    [
      'cauchy(2, 3) ppf(0) and ppf(1)',
      () => T.stack([D.cauchy(2, 3).ppf(0), D.cauchy(2, 3).ppf(1)]),
      [-Infinity, Infinity],
    ],
  ]
  it('keep the relative precision of the Cauchy quantile and tails', () => {
    expectRows(tails, 1e-12)
    // cdf and sf in the far tails come back to 1e-14 with a relative (not absolute) error of rounding.
    const x = -31830988618379.066
    for (const got of [T.item(D.cauchy(0, 1).cdf(x)), T.item(D.cauchy(0, 1).sf(-x))])
      expect(Math.abs(got - 1e-14) / 1e-14).toBeLessThan(1e-12)
  })
})
