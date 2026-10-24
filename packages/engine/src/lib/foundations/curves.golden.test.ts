import { goldenSuite } from '../../testing/suite.ts'
import { add, cos, div, mul, neg, sin, stack, sub, type Tensor, tensor, toFlat } from '../tensor/index.ts'
import * as F from './index.ts'

const CURVES: Readonly<Record<string, F.ParametricCurve>> = {
  helix: (t) => stack([mul(cos(t), 2), mul(sin(t), 2), div(t, 2)], -1),
  twisted: (t) => stack([t, mul(t, t), mul(t, mul(t, t))], -1),
  trefoil: (t) =>
    stack([add(sin(t), mul(sin(mul(t, 2)), 2)), sub(cos(t), mul(cos(mul(t, 2)), 2)), neg(sin(mul(t, 3)))], -1),
  ellipse: (t) => stack([mul(cos(t), 3), sin(t)], -1),
  cardioid: (t) => {
    const r = sub(1, cos(t))
    return stack([mul(r, cos(t)), mul(r, sin(t))], -1)
  },
}

const curve = (o: Readonly<Record<string, unknown>>): F.ParametricCurve => {
  const c = CURVES[o['curve'] as string]
  if (c === undefined) throw new Error(`no curve ${String(o['curve'])}`)
  return c
}
const conic = (c: Tensor): F.Conic => F.classifyConic(toFlat(c) as [number, number, number, number, number, number])
const sorted = (ps: readonly (readonly number[])[]): Tensor =>
  tensor([...ps].map((p) => [...p]).sort((a, b) => (a[0] ?? 0) - (b[0] ?? 0) || (a[1] ?? 0) - (b[1] ?? 0)))

goldenSuite('curves against sympy, scipy.integrate and scipy.interpolate.BPoly', 'foundations/curves', {
  arcLength: (_, o) => F.arcLength(curve(o), 0, 2, { epsabs: 1e-12, epsrel: 1e-12 }),
  curvature: ([t], o) => F.curvature(curve(o), t),
  torsion: ([t], o) => F.torsion(curve(o), t),
  tangent: ([t], o) => tensor(F.frenetFrame(curve(o), t).tangent),
  normal: ([t], o) => tensor(F.frenetFrame(curve(o), t).normal),
  conicFoci: ([c]) => sorted(conic(c).foci),
  conicEccentricity: ([c]) => conic(c).eccentricity,
  conicCentre: ([c]) => tensor(conic(c).centre ?? []),
  conicVertex: ([c]) => tensor(conic(c).vertices[0] ?? []),
  conicFocalParameter: ([c]) => conic(c).semiAxes[0] ?? Number.NaN,
  conicType: ([c]) => conic(c).type,
  bezier: ([control, t]) => F.bezier(control, t),
  bezierDerivative: ([control, t]) => F.bezier(F.bezierDerivative(rows(control)), t),
})

function rows(t: Tensor): number[][] {
  const f = toFlat(t)
  const c = t.shape[1] ?? 1
  return Array.from({ length: t.shape[0] ?? 0 }, (_, i) => f.slice(i * c, i * c + c))
}
