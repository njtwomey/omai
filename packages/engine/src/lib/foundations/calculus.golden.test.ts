import { goldenSuite, type Params } from '../../testing/suite.ts'
import {
  abs,
  add,
  cos,
  div,
  exp,
  expm1,
  item,
  log,
  log1p,
  mul,
  neg,
  sin,
  sub,
  tensor,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import * as F from './index.ts'

const sq = (x: Value): Value => mul(x, x)

const LIMITS: Readonly<Record<string, F.RealFunction>> = {
  sinc: (x) => div(sin(x), x),
  compound: (x) => exp(div(log1p(x), x)),
  expm1: (x) => div(expm1(x), x),
  hole: (x) => div(sub(sq(x), 1), sub(x, 1)),
  versine: (x) => div(sub(1, cos(x)), sq(x)),
  signRight: (x) => div(abs(x), x),
  signLeft: (x) => div(abs(x), x),
  euler: (x) => exp(mul(x, log(add(1, div(1, x))))),
  rational: (x) => div(add(mul(3, sq(x)), x), sub(sq(x), 5)),
}

const FUNCTIONS: Readonly<Record<string, F.RealFunction>> = {
  cubic: (x) => sub(mul(x, sq(x)), mul(3, x)),
  quartic: (x) => add(sub(sq(sq(x)), mul(2, sq(x))), mul(x, 0.5)),
  wave: (x) => add(sin(x), div(x, 3)),
  expsin: (x) => exp(sin(x)),
  log1p: (x) => log1p(x),
  gauss: (x) => exp(neg(sq(x))),
}

const fn = (table: Readonly<Record<string, F.RealFunction>>, o: Params): F.RealFunction => {
  const f = table[o['f'] as string]
  if (f === undefined) throw new Error(`no function ${String(o['f'])}`)
  return f
}

goldenSuite('calculus against sympy, scipy.integrate and scipy.optimize', 'foundations/calculus', {
  limit: ([a], o) => {
    const side = o['side'] === '+' ? 'right' : o['side'] === '-' ? 'left' : 'both'
    return F.limit(fn(LIMITS, o), item(a), { side }).value
  },
  riemann: (_, o) => F.riemannSum(fn(FUNCTIONS, o), -1, 2, 7, o['rule'] as F.RiemannRule).value,
  taylor: ([c], o) => tensor([...F.taylorApproximation(fn(FUNCTIONS, o), item(c), 5).coefficients]),
  derivative: ([c], o) =>
    F.derivatives(fn(FUNCTIONS, o), [item(c)], o['k'] as number)[o['k'] as number]?.[0] ?? Number.NaN,
  newton: ([x0], o) => F.newtonSteps(fn(FUNCTIONS, o), item(x0)).root,
  critical: (_, o) => tensor(F.criticalPoints(fn(FUNCTIONS, o), -3, 3).map((c) => c.x)),
  criticalKinds: (_, o) =>
    F.criticalPoints(fn(FUNCTIONS, o), -3, 3)
      .map((c) => c.kind)
      .join(','),
  meanValue: ([ab], o) => {
    const [a = 0, b = 0] = toFlat(ab)
    return tensor(F.meanValuePoints(fn(FUNCTIONS, o), a, b).points)
  },
})
