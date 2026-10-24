import { goldenSuite, type Params } from '../../testing/suite.ts'
import { abs, add, cos, mul, tensor, toFlat, type Value } from '../tensor/index.ts'
import * as F from './index.ts'

const factorial = (m: number): number => {
  let f = 1
  for (let i = 2; i <= m; i++) f *= i
  return f
}

const TERMS: Readonly<Record<string, F.Sequence>> = {
  basel: (k) => 1 / (k + 1) ** 2,
  alternatingHarmonic: (k) => (-1) ** k / (k + 1),
  leibniz: (k) => (-1) ** k / (2 * k + 1),
  exp2: (k) => 2 ** k / factorial(k),
  alternatingSquares: (k) => (-1) ** k / (k + 1) ** 2,
  geometric: (k) => (2 / 3) ** k,
}
const RATIOS: Readonly<Record<string, F.Sequence>> = {
  polyOverExp: (n) => (n * n + 1) / 2 ** n,
  expOverPoly: (n) => 3 ** n / n ** 4,
  fifthOverThird: (n) => n ** 5 / 3 ** n,
}
const ROOTS: Readonly<Record<string, F.Sequence>> = {
  half: (n) => (n / (2 * n + 1)) ** n,
  third: (n) => (n / (3 * n + 2)) ** n,
}
// Logarithms of the coefficients, so that no power overflows.
const RADII: Readonly<Record<string, F.Sequence>> = {
  two: (n) => Math.log(n + 1) - n * Math.LN2,
  five: (n) => 2 * Math.log(n) - n * Math.log(5),
  three: (n) => -Math.log(n) - n * Math.log(3),
}
const FOURIER: Readonly<Record<string, (x: Value) => Value>> = {
  ramp: (x) => x,
  parabola: (x) => mul(x, x),
  abs: (x) => abs(x),
  mixed: (x) => add(x, cos(mul(x, 2))),
}

const pick = <T>(table: Readonly<Record<string, T>>, key: unknown): T => {
  const v = table[key as string]
  if (v === undefined) throw new Error(`unknown ${String(key)}`)
  return v
}
const fourier = (o: Params): F.FourierCoefficients =>
  F.fourierCoefficients(pick(FOURIER, o['f']), 2 * Math.PI, 5, { points: [0] })

goldenSuite('series against sympy, numpy and scipy.optimize', 'foundations/series', {
  partialSums: (_, o) => tensor(F.partialSums(F.sequenceTerms(pick(TERMS, o['terms']), 12))),
  seriesSum: (_, o) => F.seriesSum(pick(TERMS, o['terms']), 40).accelerated,
  ratioTest: (_, o) => F.ratioTest(pick(RATIOS, o['terms'])).statistic,
  rootTest: (_, o) => F.rootTest(pick(ROOTS, o['terms'])).statistic,
  radius: (_, o) => F.radiusOfConvergence(pick(RADII, o['coefficients']), { logAbs: true }).root,
  fourierA: (_, o) => tensor(fourier(o).a),
  fourierB: (_, o) => tensor(fourier(o).b),
  gibbsPeak: (_, o) => {
    const n = o['terms'] as number
    return F.gibbsOvershoot(F.squareWaveCoefficients(n), n, { at: 0, left: -1, right: 1 }).peak
  },
  rationalSeries: ([p, q]) => tensor(F.rationalSeries(toFlat(p), toFlat(q), 12)),
  seriesProduct: ([a, b]) => tensor(F.seriesProduct(toFlat(a), toFlat(b))),
})
