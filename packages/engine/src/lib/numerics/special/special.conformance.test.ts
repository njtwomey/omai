import { describe, expect, it } from 'vitest'
import { checkPrimitive, grad } from '../../autodiff/index.ts'
import { item, tensor, type Value } from '../../tensor/index.ts'
import * as S from './index.ts'

// Every special function passes the primitive conformance test at interior points: reverse mode against central
// differences, forward mode against central differences, and the dot-product test linking the two.
const cases: readonly (readonly [string, (...xs: Value[]) => Value, readonly number[][]])[] = [
  ['erf', S.erf, [[-1.2, 0.3, 2]]],
  ['erfc', S.erfc, [[-1.2, 0.3, 2]]],
  ['erfcx', S.erfcx, [[-1.2, 0.3, 4]]],
  ['erfinv', S.erfinv, [[-0.7, 0.1, 0.9]]],
  ['erfcinv', S.erfcinv, [[0.1, 1, 1.8]]],
  ['dawsn', S.dawsn, [[-2, 0.5, 7]]],
  ['ndtr', S.ndtr, [[-3, 0, 2]]],
  ['logNdtr', S.logNdtr, [[-10, 0, 3]]],
  ['ndtri', S.ndtri, [[0.01, 0.5, 0.97]]],
  [
    'owensT',
    S.owensT,
    [
      [-1, 0.5, 2],
      [0.3, -2, 3],
    ],
  ],
  [
    'logIntervalProbability',
    S.logIntervalProbability,
    [
      [-2, 0.5, 4],
      [-1, 1, 5],
    ],
  ],
  ['logit', S.logit, [[0.1, 0.5, 0.8]]],
  ['log1mexp', S.log1mexp, [[-3, -0.5, -0.01]]],
  [
    'logaddexp',
    S.logaddexp,
    [
      [-1, 2, 0],
      [3, -1, 0.5],
    ],
  ],
  [
    'xlogy',
    S.xlogy,
    [
      [0.5, 2, -1],
      [0.3, 4, 2],
    ],
  ],
  [
    'relEntr',
    S.relEntr,
    [
      [0.5, 2, 1],
      [0.3, 4, 2],
    ],
  ],
  ['gamma', S.gamma, [[0.4, 2.5, -1.5]]],
  ['lgamma', S.lgamma, [[0.4, 2.5, 30]]],
  ['digamma', S.digamma, [[0.4, 2.5, -1.5]]],
  ['trigamma', S.trigamma, [[0.4, 2.5, 30]]],
  [
    'lbeta',
    S.lbeta,
    [
      [0.5, 3, 200],
      [2, 0.7, 1e-2],
    ],
  ],
  [
    'lbinom',
    S.lbinom,
    [
      [10, 20.5, 1e5],
      [3, 4.5, 7],
    ],
  ],
  [
    'gammainc',
    S.gammainc,
    [
      [0.5, 3, 40],
      [0.2, 4, 35],
    ],
  ],
  [
    'gammaincc',
    S.gammaincc,
    [
      [0.5, 3, 40],
      [2, 1, 50],
    ],
  ],
  [
    'logGammainc',
    S.logGammainc,
    [
      [0.5, 3, 40],
      [0.2, 0.01, 35],
    ],
  ],
  [
    'logGammaincc',
    S.logGammaincc,
    [
      [0.5, 3, 40],
      [2, 60, 50],
    ],
  ],
  [
    'gammaincinv',
    S.gammaincinv,
    [
      [0.5, 3, 40],
      [0.2, 0.5, 0.9],
    ],
  ],
  [
    'gammainccinv',
    S.gammainccinv,
    [
      [0.5, 3, 40],
      [0.2, 1e-3, 0.9],
    ],
  ],
  [
    'betainc',
    S.betainc,
    [
      [0.5, 3, 40],
      [2, 0.7, 30],
      [0.3, 0.6, 0.5],
    ],
  ],
  [
    'betaincinv',
    S.betaincinv,
    [
      [0.5, 3, 40],
      [2, 0.7, 30],
      [0.3, 0.6, 0.5],
    ],
  ],
  [
    'stdtr',
    S.stdtr,
    [
      [1.5, 4, 300],
      [-2, 0.1, 3],
    ],
  ],
  [
    'stdtrit',
    S.stdtrit,
    [
      [1.5, 4, 300],
      [0.05, 0.6, 0.99],
    ],
  ],
  [
    'chdtr',
    S.chdtr,
    [
      [1, 4, 30],
      [0.5, 3, 40],
    ],
  ],
  [
    'fdtr',
    S.fdtr,
    [
      [1, 4, 30],
      [2, 5, 10],
      [0.5, 1.5, 3],
    ],
  ],
  ['kolmogorov', S.kolmogorov, [[0.3, 0.9, 2]]],
  ['zeta', S.zeta, [[1.5, 3, 12]]],
  [
    'hurwitzZeta',
    S.hurwitzZeta,
    [
      [1.5, 3, 12],
      [0.5, 2, 7],
    ],
  ],
  ['lambertW', S.lambertW, [[-0.2, 1, 50]]],
  ['expi', S.expi, [[-2, 0.5, 3]]],
  [
    'iv',
    S.iv,
    [
      [0, 1.5, 4],
      [0.5, 3, 6],
    ],
  ],
  [
    'ive',
    S.ive,
    [
      [0, 1.5, 4],
      [0.5, 3, 200],
    ],
  ],
  [
    'logIv',
    S.logIv,
    [
      [0, 1.5, 4],
      [0.5, 3, 2000],
    ],
  ],
  [
    'kv',
    S.kv,
    [
      [0, 1.5, 4],
      [0.5, 3, 6],
    ],
  ],
  [
    'jv',
    S.jv,
    [
      [0, 1.5, 4],
      [0.5, 3, 20],
    ],
  ],
  [
    'yv',
    S.yv,
    [
      [0, 1.5, 4],
      [0.5, 3, 20],
    ],
  ],
  ['ellipk', S.ellipk, [[-2, 0.3, 0.9]]],
  ['ellipe', S.ellipe, [[-2, 0.3, 0.9]]],
  [
    'ellipkinc',
    S.ellipkinc,
    [
      [0.3, 1.2, 4],
      [-2, 0.3, 0.8],
    ],
  ],
  [
    'ellipeinc',
    S.ellipeinc,
    [
      [0.3, 1.2, 4],
      [-2, 0.3, 0.8],
    ],
  ],
  [
    'ellipj.sn',
    (u, m) => S.ellipj(u as Value, m as Value).sn,
    [
      [0.3, 1.2, 4],
      [0.1, 0.3, 0.8],
    ],
  ],
]

// The Bessel and elliptic evaluators carry about 1e-9 of error in their gradients, which differencing the gradient
// amplifies: their second derivatives are checked to 1e-2, still enough to catch a wrong sign or factor.
const looseSecondOrder = new Set(['iv', 'ive', 'logIv', 'kv', 'jv', 'yv', 'ellipj.sn'])

describe('special functions are conformant primitives', () => {
  for (const [name, f, inputs] of cases) {
    it(name, () => {
      const xs = inputs.map((x) => tensor(x))
      const loose = looseSecondOrder.has(name)
      expect(checkPrimitive(f, xs, { rtol: 1e-5, atol: 1e-7, ...(loose ? {} : { order: 2 }) }).failures).toEqual([])
      if (loose) {
        expect(checkPrimitive(f, xs, { rtol: 1e-2, atol: 1e-4, order: 2, vmap: false }).failures).toEqual([])
      }
    })
  }
})

describe('second derivatives', () => {
  it('of the gamma CDF in x are the derivative of the density', () => {
    // d^2 P / dx^2 = density' = density ((a - 1) / x - 1).
    const a = 3
    const x = 2
    const d2 = item(grad(grad((v: Value) => S.gammainc(a, v)))(tensor(x)) as Value)
    const density = (x ** (a - 1) * Math.exp(-x)) / 2
    expect(d2).toBeCloseTo(density * ((a - 1) / x - 1), 12)
  })

  it('of lgamma are trigamma', () => {
    expect(item(grad(grad((v: Value) => S.lgamma(v)))(tensor(2.5)) as Value)).toBeCloseTo(item(S.trigamma(2.5)), 12)
  })

  it('of the Student t quantile in p match finite differences', () => {
    const h = 1e-5
    const f = (p: number): number => item(grad((v: Value) => S.stdtrit(5, v))(tensor(p)) as Value)
    const d2 = item(grad(grad((v: Value) => S.stdtrit(5, v)))(tensor(0.8)) as Value)
    expect(d2).toBeCloseTo((f(0.8 + h) - f(0.8 - h)) / (2 * h), 4)
  })
})
