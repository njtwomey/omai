import { goldenSuite, type Params } from '../../testing/suite.ts'
import {
  cos,
  cosh,
  mul,
  sin,
  sliceAxis,
  squeeze,
  stack,
  sub,
  type Tensor,
  tensor,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import * as F from './index.ts'

const uv = (p: Value): [Value, Value] => [squeeze(sliceAxis(p, -1, 0, 1), -1), squeeze(sliceAxis(p, -1, 1, 2), -1)]

const SURFACES: Readonly<Record<string, F.ParametricSurface>> = {
  sphere: F.sphere(2),
  torus: F.torus(3, 1),
  saddle: F.graphSurface((p) => {
    const [x, y] = uv(p)
    return mul(x, y)
  }),
  monkey: F.graphSurface((p) => {
    const [x, y] = uv(p)
    return sub(mul(x, mul(x, x)), mul(3, mul(x, mul(y, y))))
  }),
  catenoid: (p) => {
    const [a, b] = uv(p)
    return stack([mul(cosh(a), cos(b)), mul(cosh(a), sin(b)), a], -1)
  },
  helicoid: (p) => {
    const [a, b] = uv(p)
    return stack([mul(a, cos(b)), mul(a, sin(b)), mul(b, 0.5)], -1)
  },
}

const surface = (o: Params): F.ParametricSurface => {
  const s = SURFACES[o['surface'] as string]
  if (s === undefined) throw new Error(`no surface ${String(o['surface'])}`)
  return s
}
const pair = (t: Tensor): [number, number] => toFlat(t) as [number, number]

goldenSuite('surfaces against sympy and scipy.integrate.solve_ivp', 'foundations/surfaces', {
  E: ([p], o) => F.firstFundamentalForm(surface(o), p).uu,
  F: ([p], o) => F.firstFundamentalForm(surface(o), p).uv,
  G: ([p], o) => F.firstFundamentalForm(surface(o), p).vv,
  L: ([p], o) => F.secondFundamentalForm(surface(o), p).uu,
  M: ([p], o) => F.secondFundamentalForm(surface(o), p).uv,
  N: ([p], o) => F.secondFundamentalForm(surface(o), p).vv,
  K: ([p], o) => F.gaussianCurvature(surface(o), p),
  H: ([p], o) => F.meanCurvature(surface(o), p),
  principal: ([p], o) => {
    const rows = (toFlat(p).length / 2) | 0
    const flat = toFlat(p)
    return tensor(
      Array.from(
        { length: rows },
        (_, i) => F.principalCurvatures(surface(o), [flat[2 * i] ?? 0, flat[2 * i + 1] ?? 0]).curvatures,
      ),
    )
  },
  christoffel: ([p], o) => tensor(F.christoffelSymbols(surface(o), pair(p))),
  geodesic: ([p, vel], o) => {
    const g = F.geodesic(surface(o), pair(p), pair(vel), { length: 1, steps: 400 })
    return tensor(g.parameters[g.parameters.length - 1] ?? [])
  },
  area: ([box], o) => {
    const [a, b] = pair(box)
    return F.surfaceArea(surface(o), [0, a], [0, b])
  },
})
