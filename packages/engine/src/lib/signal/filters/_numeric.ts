// Scalar numerics for filter design that the domains below signal do not provide in the form scipy uses: the complete elliptic
// integral with the complementary parameter (scipy.special.ellipkm1, by the arithmetic-geometric mean), the
// inverse Jacobi functions of scipy's elliptic design by Landen transformations, the bounded scalar minimiser of
// scipy.optimize.fminbound and the secant iteration of scipy.optimize.newton (both the library's, from `optim` and
// `numerics/roots`, under scipy's argument names).
import { MdmlError } from '../../base/index.ts'
import { complexOps as C, type ComplexNumber } from '../../numerics/differentiate/index.ts'
import { newton } from '../../numerics/roots/index.ts'
import { ellipk } from '../../numerics/special/index.ts'
import { fminbound as fminboundLibrary } from '../../optim/index.ts'
import { item } from '../../tensor/index.ts'

function agm(a0: number, b0: number): number {
  let a = a0
  let b = b0
  for (let i = 0; i < 60; i++) {
    const an = 0.5 * (a + b)
    const bn = Math.sqrt(a * b)
    if (Math.abs(an - bn) <= 1e-16 * an) return an
    a = an
    b = bn
  }
  return 0.5 * (a + b)
}

/** K(1 - p) without cancellation for small p (scipy.special.ellipkm1). */
export function ellipKm1(p: number): number {
  if (p <= 0) return Number.POSITIVE_INFINITY
  return Math.PI / (2 * agm(1, Math.sqrt(p)))
}

/** Principal complex arcsine: -i log(i z + sqrt(1 - z^2)). */
function casin(z: ComplexNumber): ComplexNumber {
  const iz = C.of(-z.im, z.re)
  const root = C.sqrt(C.sub(C.of(1), C.mul(z, z)))
  const l = C.log(C.add(iz, root))
  return C.of(l.im, -l.re)
}

/** The inverse Jacobi sn of a complex argument by Landen transformations (scipy's _arc_jac_sn). */
function arcJacSn(w: ComplexNumber, m: number): ComplexNumber {
  const complement = (kx: ComplexNumber): ComplexNumber => C.sqrt(C.mul(C.sub(C.of(1), kx), C.add(C.of(1), kx)))
  const k = Math.sqrt(m)
  if (k > 1) return C.of(Number.NaN)
  if (k === 1) {
    // arctanh(w) = 0.5 log((1 + w) / (1 - w))
    return C.mul(C.log(C.div(C.add(C.of(1), w), C.sub(C.of(1), w))), 0.5)
  }
  const ks: number[] = [k]
  let niter = 0
  while (ks[ks.length - 1] !== 0) {
    const kk = ks[ks.length - 1] ?? 0
    const kp = Math.sqrt((1 - kk) * (1 + kk))
    ks.push((1 - kp) / (1 + kp))
    niter++
    if (niter > 10) throw new MdmlError('convergence', 'ellip: the Landen transformation does not converge')
  }
  let kProd = Math.PI / 2
  for (const v of ks.slice(1)) kProd *= 1 + v
  let wn = w
  for (let i = 0; i + 1 < ks.length; i++) {
    const kn = ks[i] ?? 0
    const knext = ks[i + 1] ?? 0
    wn = C.div(C.mul(wn, 2), C.mul(C.add(C.of(1), complement(C.mul(wn, kn))), 1 + knext))
  }
  return C.mul(casin(wn), (2 / Math.PI) * kProd)
}

/** The real inverse Jacobi sc function, from arcsn of an imaginary argument (scipy's _arc_jac_sc1). */
export function arcJacSc1(w: number, m: number): number {
  const z = arcJacSn(C.of(0, w), m)
  if (Math.abs(z.re) > 1e-14) throw new MdmlError('value', 'ellip: arc sc did not stay imaginary')
  return z.im
}

/** The degree equation of elliptic filters by nome series (scipy's _ellipdeg). */
export function ellipDeg(n: number, m1: number): number {
  const k1 = item(ellipk(m1))
  const k1p = ellipKm1(m1)
  const q1 = Math.exp((-Math.PI * k1p) / k1)
  const q = q1 ** (1 / n)
  let num = 0
  for (let j = 0; j <= 7; j++) num += q ** (j * (j + 1))
  let den = 0
  for (let j = 1; j <= 8; j++) den += q ** (j * j)
  return 16 * q * (num / (1 + 2 * den)) ** 4
}

/** Bounded Brent minimisation, scipy.optimize.fminbound (xatol 1e-5, 500 evaluations): the library's, from `optim`. */
export function fminbound(f: (x: number) => number, x1: number, x2: number, xatol = 1e-5, maxfun = 500): number {
  return fminboundLibrary(f, { bounds: [x1, x2], tol: xatol, maxIterations: maxfun }).x
}

/** The secant iteration of scipy.optimize.newton without a derivative (tol 1.48e-8, 50 iterations): the library's. */
export function secant(f: (x: number) => number, x0: number, tol = 1.48e-8, maxiter = 50): number {
  const r = newton(f, x0, { tol, maxIterations: maxiter })
  if (r.flag === 'maximum iterations') throw new MdmlError('convergence', 'secant: no convergence')
  return r.root
}
