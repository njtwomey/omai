import { goldenSuite, type Params } from '../../testing/suite.ts'
import type { ComplexNumber } from '../numerics/differentiate/index.ts'
import {
  add,
  type ComplexValue,
  cDiv,
  cExp,
  cMul,
  complex,
  mul,
  onesLike,
  stack,
  sub,
  type Tensor,
  tensor,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import * as F from './index.ts'

const parts = (w: ComplexValue): Value => stack([w.re, w.im])
const z = (re: Tensor, im: Tensor): ComplexValue => complex(re, im)
const one = (w: ComplexValue): ComplexValue => complex(onesLike(w.re))
const numbers = (t: Tensor): ComplexNumber[] => {
  const v = toFlat(t)
  const n = v.length / 2
  return Array.from({ length: n }, (_, i) => ({ re: v[i] ?? 0, im: v[n + i] ?? 0 }))
}
const sortedParts = (w: readonly ComplexNumber[]): Tensor => {
  const s = [...w].sort((p, q) => p.re - q.re || p.im - q.im)
  return tensor([s.map((p) => p.re), s.map((p) => p.im)])
}

const DERIVATIVES: Readonly<Record<string, F.ComplexFunction>> = {
  square: F.conformalMaps.square,
  exp: F.conformalMaps.exp,
  joukowski: F.conformalMaps.joukowski,
  cayley: F.conformalMaps.cayley,
}
const shift = (w: ComplexValue, c: number): ComplexValue => ({ re: sub(w.re, c), im: w.im })
const RESIDUES: Readonly<Record<string, F.ComplexFunction>> = {
  poles: (w) => cDiv(one(w), cMul(w, shift(w, 1))),
  double: (w) => cDiv(cExp(w), cMul(w, w)),
  cotangent: (w) => cDiv(F.complexCos(w), F.complexSin(w)),
  rational: (w) => {
    const w2 = cMul(w, w)
    return cDiv({ re: add(w2.re, 1), im: w2.im }, { re: sub(cMul(w2, w).re, 8), im: cMul(w2, w).im })
  },
}
const pick = <T>(table: Readonly<Record<string, T>>, o: Params): T => {
  const v = table[o['f'] as string]
  if (v === undefined) throw new Error(`unknown ${String(o['f'])}`)
  return v
}

goldenSuite('complex analysis against numpy, scipy and sympy', 'foundations/complex', {
  log: ([re, im]) => parts(F.complexLog(z(re, im))),
  sqrt: ([re, im]) => parts(F.complexSqrt(z(re, im))),
  sin: ([re, im]) => parts(F.complexSin(z(re, im))),
  cos: ([re, im]) => parts(F.complexCos(z(re, im))),
  pow: ([re, im, ure, uim]) => parts(F.complexPow(z(re, im), z(ure as Tensor, uim as Tensor))),
  polar: ([re, im]) => {
    const p = F.toPolarForm(z(re, im))
    return stack([p.r, p.theta])
  },
  nthRoots: ([spec]) => {
    const [re = 0, im = 0, n = 1] = toFlat(spec)
    const r = F.nthRoots({ re, im }, n)
    const re2 = toFlat(r.re)
    const im2 = toFlat(r.im)
    return sortedParts(re2.map((x, i) => ({ re: x, im: im2[i] ?? 0 })))
  },
  domainModulus: () =>
    tensor(
      F.domainColouring(
        domainF,
        [
          [-2, 2],
          [-1.5, 1.5],
        ],
        { size: [7, 5] },
      ).modulus,
    ),
  domainArgument: () =>
    tensor(
      F.domainColouring(
        domainF,
        [
          [-2, 2],
          [-1.5, 1.5],
        ],
        { size: [7, 5] },
      ).argument,
    ),
  mobiusApply: ([coef, re, im]) => parts(F.mobiusApply(mobiusOf(coef), z(re, im as Tensor))),
  mobiusFixed: ([coef]) => sortedParts(F.mobiusFixedPoints(mobiusOf(coef)).points),
  mobiusFromPoints: ([src, dst, probe]) => {
    const [z1, z2, z3] = numbers(src) as [ComplexNumber, ComplexNumber, ComplexNumber]
    const [w1, w2, w3] = numbers(dst) as [ComplexNumber, ComplexNumber, ComplexNumber]
    const m = F.mobiusFromPoints([z1, z2, z3], [w1, w2, w3])
    const ps = numbers(probe as Tensor)
    return parts(
      F.mobiusApply(
        m,
        complex(
          ps.map((p) => p.re),
          ps.map((p) => p.im),
        ),
      ),
    )
  },
  derivative: ([p], o) => {
    const [re = 0, im = 0] = toFlat(p)
    const d = F.cauchyRiemann(pick(DERIVATIVES, o), { re, im }).derivative
    return tensor([d.re, d.im])
  },
  residue: ([p], o) => {
    const [re = 0, im = 0] = toFlat(p)
    const r = F.residueAt(pick(RESIDUES, o), { re, im })
    return tensor([r.re, r.im])
  },
  laurentExp: () => tensor([0, 1, 2, 3, 4, 5].map((k) => F.laurentCoefficient(cExp, 0, k, { radius: 1 }).re)),
  contourParabola: () => {
    const c = F.contourIntegral(
      (w) => cMul(w, w),
      (t) => complex(t, mul(t, t)),
      0,
      1,
      { method: 'quad' },
    )
    return tensor([c.re, c.im])
  },
  argumentPrinciple: ([poly, radius]) => {
    const coeffs = numbers(poly)
    const f: F.ComplexFunction = (w) => {
      let acc: ComplexValue = complex(mul(w.re, 0))
      for (const c of coeffs) acc = add2(cMul(acc, w), complex(add(mul(w.re, 0), c.re), add(mul(w.re, 0), c.im)))
      return acc
    }
    return F.argumentPrinciple(f, F.circleContour(0, toFlat(radius as Tensor)[0] ?? 1)).count
  },
})

function add2(p: ComplexValue, q: ComplexValue): ComplexValue {
  return { re: add(p.re, q.re), im: add(p.im, q.im) }
}

function domainF(w: ComplexValue): ComplexValue {
  const w2 = cMul(w, w)
  return cDiv({ re: sub(w2.re, 1), im: w2.im }, { re: w2.re, im: add(w2.im, 1) })
}

function mobiusOf(coef: Tensor): F.Mobius {
  const [ar = 0, ai = 0, br = 0, bi = 0, cr = 0, ci = 0, dr = 0, di = 0] = toFlat(coef)
  return F.mobius({ re: ar, im: ai }, { re: br, im: bi }, { re: cr, im: ci }, { re: dr, im: di })
}
