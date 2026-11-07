import { goldenSuite, type Params } from '../../testing/suite.ts'
import {
  add,
  cos,
  div,
  exp,
  mul,
  neg,
  sin,
  sliceAxis,
  squeeze,
  stack,
  sub,
  tensor,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import * as F from './index.ts'

const c = (p: Value, i: number): Value => squeeze(sliceAxis(p, -1, i, i + 1), -1)
const sq = (a: Value): Value => mul(a, a)

const SCALARS: Readonly<Record<string, F.ScalarField>> = {
  bowl: (p) => add(sq(c(p, 0)), mul(3, sq(c(p, 1)))),
  rosenbrock: (p) => add(sq(sub(1, c(p, 0))), mul(100, sq(sub(c(p, 1), sq(c(p, 0)))))),
  wave: (p) => add(mul(sin(c(p, 0)), cos(c(p, 1))), div(mul(c(p, 0), c(p, 1)), 4)),
  gauss3: (p) => add(exp(sub(sub(neg(sq(c(p, 0))), mul(2, sq(c(p, 1)))), div(sq(c(p, 2)), 2))), mul(c(p, 0), c(p, 2))),
}

const FIELDS: Readonly<Record<string, F.PointVectorField>> = {
  swirl: (p) => {
    const [x, y, z] = [c(p, 0), c(p, 1), c(p, 2)]
    return stack([add(neg(y), mul(x, z)), add(x, mul(y, z)), sub(sq(z), mul(x, y))], -1)
  },
  radial: (p) => {
    const [x, y, z] = [c(p, 0), c(p, 1), c(p, 2)]
    return stack([mul(x, add(add(sq(x), sq(y)), sq(z))), y, mul(z, exp(x))], -1)
  },
  plane: (p) => {
    const [x, y] = [c(p, 0), c(p, 1)]
    return stack([sub(sq(x), y), mul(x, sq(y))], -1)
  },
}

const PROBLEMS: Readonly<Record<string, readonly [F.ScalarField, F.ScalarField]>> = {
  xyCircle: [(p) => mul(c(p, 0), c(p, 1)), (p) => sub(add(sq(c(p, 0)), sq(c(p, 1))), 2)],
  sumEllipse: [(p) => add(c(p, 0), mul(2, c(p, 1))), (p) => sub(add(sq(c(p, 0)), mul(4, sq(c(p, 1)))), 8)],
  distanceLine: [(p) => add(sq(sub(c(p, 0), 3)), sq(sub(c(p, 1), 1))), (p) => sub(add(c(p, 0), c(p, 1)), 1)],
}

const pick = <T>(table: Readonly<Record<string, T>>, key: unknown): T => {
  const v = table[key as string]
  if (v === undefined) throw new Error(`unknown ${String(key)}`)
  return v
}
const helix: F.ParametricCurve = (t) => stack([cos(t), sin(t), mul(t, 0.5)], -1)
const patch: F.ParametricSurface = (uv) => stack([c(uv, 0), c(uv, 1), add(sq(c(uv, 0)), sq(c(uv, 1)))], -1)
const unitBox: F.ParameterBox = [
  [0, 1],
  [0, 1],
]

goldenSuite('multivariable calculus against sympy', 'foundations/multivariable', {
  gradient: ([p], o: Params) => F.gradientAt(pick(SCALARS, o['f']), p),
  hessian: ([p], o) => F.hessianAt(pick(SCALARS, o['f']), p),
  directional: ([p, d], o) => F.directionalDerivative(pick(SCALARS, o['f']), p, d),
  jacobian: ([p], o) => F.jacobianAt(pick(FIELDS, o['field']), p),
  divergence: ([p], o) => F.divergence(pick(FIELDS, o['field']), p),
  curl: ([p], o) => F.curl(pick(FIELDS, o['field']), p),
  lagrange: ([start], o) => {
    const [f, g] = pick(PROBLEMS, o['problem'])
    const r = F.lagrangeMultipliers(f, [g], toFlat(start))
    return tensor([...r.x, ...r.multipliers])
  },
  scalarLine: () => F.scalarLineIntegral((p) => add(sq(c(p, 0)), c(p, 2)), helix, 0, 2),
  vectorLine: () => F.vectorLineIntegral(pick(FIELDS, 'swirl'), helix, 0, 2),
  scalarSurface: () => F.scalarSurfaceIntegral((p) => add(mul(c(p, 0), c(p, 1)), c(p, 2)), patch, unitBox),
  surfaceFlux: () => F.surfaceFlux(pick(FIELDS, 'swirl'), patch, unitBox),
})
