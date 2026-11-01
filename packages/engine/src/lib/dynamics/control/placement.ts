/**
 * Pole placement: choosing a state-feedback gain K so that the closed-loop matrix A - B K has prescribed eigenvalues.
 * `place` follows Kautsky, Nichols and Van Dooren's method for multi-input systems (scipy.signal.place_poles), choosing
 * well-conditioned eigenvectors, and `acker` is Ackermann's formula for the single-input case.
 */

import { type Json, MdmlError } from '../../base/index.ts'
import { det, eigvals, lstsq, matrixPower, matrixRank, nullSpace, qr, solve, svd } from '../../numerics/linalg/index.ts'
import { ctrb } from '../../systems/index.ts'
import {
  add,
  asTensor,
  eye,
  matmul,
  mul,
  primalOf,
  sub,
  type Tensor,
  type TensorLike,
  transpose,
} from '../../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  numberFromJson,
  numbersFromJson,
  numbersToJson,
  numberToJson,
  runAlgorithm,
  scalar,
  tensor,
} from '../../trace/index.ts'
import { fromNumbers, numbersOf } from '../ode/_vec.ts'

/** A pole: a real number or a complex value (its conjugate must be requested too). */
export type Pole = number | { readonly re: number; readonly im: number }

interface Cpole {
  readonly re: number
  readonly im: number
}

const asPole = (p: Pole): Cpole => (typeof p === 'number' ? { re: p, im: 0 } : p)
const isReal = (p: Cpole): boolean => p.im === 0

/** scipy's ordering: real poles ascending, then each complex pair with the negative imaginary part first. */
function orderPoles(poles: readonly Pole[]): Cpole[] {
  const ps = poles.map(asPole)
  const real = ps.filter(isReal).sort((a, b) => a.re - b.re)
  const lower = ps.filter((p) => p.im < 0).sort((a, b) => a.re - b.re || a.im - b.im)
  const out: Cpole[] = [...real]
  for (const p of lower) {
    if (!ps.some((q) => q.re === p.re && q.im === -p.im)) {
      throw new MdmlError('value', 'place: complex poles must come with their conjugates')
    }
    out.push(p, { re: p.re, im: -p.im })
  }
  if (out.length !== ps.length) throw new MdmlError('value', 'place: complex poles must come with their conjugates')
  return out
}

type Mat = Float64Array

const col = (m: Mat, rows: number, cols: number, j: number): Float64Array =>
  Float64Array.from({ length: rows }, (_, i) => m[i * cols + j] ?? 0)

function setCol(m: Mat, cols: number, j: number, v: ArrayLike<number>): void {
  for (let i = 0; i < v.length; i++) m[i * cols + j] = v[i] ?? 0
}

function dropCols(m: Mat, rows: number, cols: number, drop: readonly number[]): { m: Mat; cols: number } {
  const keep = Array.from({ length: cols }, (_, j) => j).filter((j) => !drop.includes(j))
  const out = new Float64Array(rows * keep.length)
  for (let i = 0; i < rows; i++)
    keep.forEach((j, k) => {
      out[i * keep.length + k] = m[i * cols + j] ?? 0
    })
  return { m: out, cols: keep.length }
}

const tensor2 = (m: Mat, r: number, c: number): Tensor => fromNumbers(m, [r, c])
const numbers2 = (t: TensorLike): Float64Array => Float64Array.from(numbersOf(primalOf(asTensor(t))))

/** An orthonormal basis of the orthogonal complement of the columns of m (n x k), as columns `[n, n - rank]`. */
function complement(m: Mat, n: number, k: number): { basis: Mat; cols: number } {
  const ns = primalOf(nullSpace(transpose(tensor2(m, n, k))))
  return { basis: Float64Array.from(ns.data), cols: ns.shape[1] ?? 0 }
}

function norm(v: ArrayLike<number>): number {
  let s = 0
  for (let i = 0; i < v.length; i++) s += (v[i] ?? 0) ** 2
  return Math.sqrt(s)
}

function allClose(v: ArrayLike<number>): boolean {
  for (let i = 0; i < v.length; i++) if (Math.abs(v[i] ?? 0) > 1e-8) return false
  return true
}

/** M v for a row-major r x c matrix. */
function mv(m: Mat, r: number, c: number, v: ArrayLike<number>): Float64Array {
  const out = new Float64Array(r)
  for (let i = 0; i < r; i++) {
    let s = 0
    for (let j = 0; j < c; j++) s += (m[i * c + j] ?? 0) * (v[j] ?? 0)
    out[i] = s
  }
  return out
}

/** The kernel K (n x d) for one pole: vectors v with u1^T (A - lambda I) v = 0. */
function kernelFor(a: Mat, u1: Mat, n: number, r: number, p: Cpole): { re: Mat; im: Mat; d: number } {
  if (isReal(p)) {
    const M = new Float64Array(r * n)
    for (let i = 0; i < r; i++) {
      for (let j = 0; j < n; j++) {
        let s = 0
        for (let k = 0; k < n; k++) s += (u1[k * r + i] ?? 0) * ((a[k * n + j] ?? 0) - (k === j ? p.re : 0))
        M[i * n + j] = s
      }
    }
    const ns = primalOf(nullSpace(tensor2(M, r, n)))
    const d = ns.shape[1] ?? 0
    return { re: Float64Array.from(ns.data), im: new Float64Array(n * d), d }
  }
  // Complex kernel through the real embedding [[Mr, -Mi], [Mi, Mr]] [xr; xi] = 0.
  const Mr = new Float64Array(r * n)
  const Mi = new Float64Array(r * n)
  for (let i = 0; i < r; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0
      for (let k = 0; k < n; k++) s += (u1[k * r + i] ?? 0) * ((a[k * n + j] ?? 0) - (k === j ? p.re : 0))
      Mr[i * n + j] = s
      Mi[i * n + j] = -p.im * (u1[j * r + i] ?? 0)
    }
  }
  const E = new Float64Array(2 * r * 2 * n)
  for (let i = 0; i < r; i++) {
    for (let j = 0; j < n; j++) {
      E[i * 2 * n + j] = Mr[i * n + j] ?? 0
      E[i * 2 * n + n + j] = -(Mi[i * n + j] ?? 0)
      E[(r + i) * 2 * n + j] = Mi[i * n + j] ?? 0
      E[(r + i) * 2 * n + n + j] = Mr[i * n + j] ?? 0
    }
  }
  const ns = primalOf(nullSpace(tensor2(E, 2 * r, 2 * n)))
  const dd = ns.shape[1] ?? 0
  const re = new Float64Array(n * dd)
  const im = new Float64Array(n * dd)
  for (let k = 0; k < dd; k++) {
    for (let i = 0; i < n; i++) {
      re[i * dd + k] = ns.data[i * dd + k] ?? 0
      im[i * dd + k] = ns.data[(n + i) * dd + k] ?? 0
    }
  }
  return { re, im, d: dd }
}

/** One KNV0 update of column j: the projection of the complement direction onto the pole's kernel. */
function knv0(X: Mat, n: number, ker: Mat, d: number, j: number): void {
  const { m: rest, cols } = dropCols(X, n, n, [j])
  const { basis, cols: c } = complement(rest, n, cols)
  const q = col(basis, n, c, c - 1)
  // y = K K^T q
  const ktq = new Float64Array(d)
  for (let a = 0; a < d; a++) for (let i = 0; i < n; i++) ktq[a] = (ktq[a] ?? 0) + (ker[i * d + a] ?? 0) * (q[i] ?? 0)
  const y = mv(ker, n, d, ktq)
  if (!allClose(y))
    setCol(
      X,
      n,
      j,
      Float64Array.from(y, (v) => v / norm(y)),
    )
}

/** One Tits-Yang update of a pair of real-pole columns (i, j) (scipy's _YT_real). */
function ytReal(X: Mat, n: number, Ki: Mat, di: number, Kj: Mat, dj: number, i: number, j: number): void {
  const { m: rest, cols } = dropCols(X, n, n, [i, j])
  const { basis, cols: c } = complement(rest, n, cols)
  const u = col(basis, n, c, c - 2)
  const v = col(basis, n, c, c - 1)
  // m = Ki^T (u v^T - v u^T) Kj
  const W = new Float64Array(n * n)
  for (let a = 0; a < n; a++)
    for (let b = 0; b < n; b++) W[a * n + b] = (u[a] ?? 0) * (v[b] ?? 0) - (v[a] ?? 0) * (u[b] ?? 0)
  const M = new Float64Array(di * dj)
  for (let a = 0; a < di; a++) {
    for (let b = 0; b < dj; b++) {
      let s = 0
      for (let p = 0; p < n; p++)
        for (let q = 0; q < n; q++) s += (Ki[p * di + a] ?? 0) * (W[p * n + q] ?? 0) * (Kj[q * dj + b] ?? 0)
      M[a * dj + b] = s
    }
  }
  const sv = svd(tensor2(M, di, dj))
  const U = numbers2(sv.u)
  const S = numbers2(sv.s)
  const Vh = numbers2(sv.vh)
  const uCols = sv.u.shape[1] ?? di
  const mu = (k: number): Float64Array => Float64Array.from({ length: di }, (_, a) => U[a * uCols + k] ?? 0)
  const nu = (k: number): Float64Array => Float64Array.from({ length: dj }, (_, b) => Vh[k * dj + b] ?? 0)
  const stackedX = Float64Array.from({ length: 2 * n }, (_, a) =>
    a < n ? (X[a * n + i] ?? 0) : (X[(a - n) * n + j] ?? 0),
  )
  let kmn: Float64Array[]
  if (Math.abs((S[0] ?? 0) - (S[1] ?? 0)) > 1e-8 * Math.max(1, Math.abs(S[0] ?? 0)) || di < 2 || dj < 2) {
    kmn = [Float64Array.from([...mv(Ki, n, di, mu(0)), ...mv(Kj, n, dj, nu(0))])]
  } else {
    kmn = [0, 1].map((k) => Float64Array.from([...mv(Ki, n, di, mu(k)), ...mv(Kj, n, dj, nu(k))]))
  }
  // t = kmn kmn^T [x_i; x_j]
  const t = new Float64Array(2 * n)
  for (const w of kmn) {
    let s = 0
    for (let a = 0; a < 2 * n; a++) s += (w[a] ?? 0) * (stackedX[a] ?? 0)
    for (let a = 0; a < 2 * n; a++) t[a] = (t[a] ?? 0) + (w[a] ?? 0) * s
  }
  const src = allClose(t) ? (kmn[0] as Float64Array) : Float64Array.from(t, (v) => (Math.SQRT2 * v) / norm(t))
  setCol(X, n, i, src.slice(0, n))
  setCol(X, n, j, src.slice(n))
}

/** scipy's update order for the Tits-Yang sweeps (zero-based pairs; (0, 0) is a KNV0 step on a real pole). */
function ytOrder(poles: readonly Cpole[]): [number, number][] {
  const nbReal = poles.filter(isReal).length
  const hnb = Math.floor(nbReal / 2)
  const first: number[] = nbReal > 0 ? [nbReal] : []
  const second: number[] = nbReal > 0 ? [1] : []
  const rComp: number[] = []
  for (let k = nbReal + 1; k <= poles.length; k += 2) rComp.push(k)
  const push = (a: number, b: number): void => {
    first.push(a)
    second.push(b)
  }
  const realStart = poles[0] !== undefined && isReal(poles[0])
  for (let p = 1; p < hnb + (nbReal % 2); p++) push(2 * p, 2 * p + 1)
  for (const k of rComp) push(k, k + 1)
  for (let p = 1; p <= hnb; p++) push(2 * p - 1, 2 * p)
  if (hnb === 0 && realStart) push(1, 1)
  for (const k of rComp) push(k, k + 1)
  for (let j = 2; j < hnb + (nbReal % 2); j++) for (let i = 1; i <= hnb; i++) push(i, i + j)
  if (hnb === 0 && realStart) push(1, 1)
  for (const k of rComp) push(k, k + 1)
  for (let j = 2; j < hnb + (nbReal % 2); j++) {
    for (let i = hnb + 1; i <= nbReal; i++) push(i, i + j > nbReal ? i + j - nbReal : i + j)
  }
  if (hnb === 0 && realStart) push(1, 1)
  for (const k of rComp) push(k, k + 1)
  for (let i = 1; i <= hnb; i++) push(i, i + hnb)
  if (hnb === 0 && realStart) push(1, 1)
  for (const k of rComp) push(k, k + 1)
  return first.map((a, k) => [a - 1, (second[k] ?? 1) - 1])
}

/** The result of {@link place}. */
export interface PlacementResult {
  /** The gain K: the closed loop is A - B K. */
  readonly gain: Tensor
  readonly requestedPoles: readonly Cpole[]
  readonly computedPoles: readonly Cpole[]
  /** The eigenvector matrix of the closed loop (its conditioning measures robustness). */
  readonly X: Tensor
  readonly iterations: number
  readonly rtol: number
}

/** The paths {@link placeAlgorithm} logs. */
export type PlacePath = 'place/conditioning' | 'place/sweep' | 'place/poles' | 'place/change'

/**
 * What pole placement logs per sweep: the condition number of the closed-loop eigenvector matrix (the robustness the
 * sweeps improve), the sweep, the relative change of its determinant (the stopping test), and on request the closed-loop
 * poles the gain gives.
 */
export const placeContract: Contract<unknown, PlacePath> = defineContract({
  'place/conditioning': scalar({ description: 'the condition number of the eigenvector matrix X', direction: 'lower' }),
  'place/sweep': scalar({ description: 'the sweep over the pole pairs, counting from 1' }),
  'place/change': scalar({ description: 'the relative change of |det X| over the sweep', direction: 'lower' }),
  'place/poles': tensor({
    role: 'output',
    shape: ['poles', 2],
    description: 'the closed-loop poles (real, imaginary) the gain gives',
  }),
})

/** Options of {@link place} and {@link placeAlgorithm}. */
export interface PlaceOptions {
  /** The update scheme: Tits and Yang's pairwise (`YT`, default) or Kautsky-Nichols-Van Dooren method 0 (`KNV0`). */
  readonly method?: 'YT' | 'KNV0'
  /** Stop when |det X| changes by less than this relative amount over a sweep (default 1e-3). */
  readonly rtol?: number
  /** The most sweeps (default 30). */
  readonly maxIterations?: number
}

/** The state of {@link placeAlgorithm}: the eigenvector matrix, the sweeps made and the last relative change of |det X|. */
export interface PlacementState {
  /** The closed-loop eigenvector matrix `[n, n]`, flat (the identity when B has full rank). */
  readonly X: Float64Array
  readonly iterations: number
  /** The relative change of |det X| in the last sweep (NaN before the first). */
  readonly rtol: number
  /** True once the change fell below the tolerance. */
  readonly stop: boolean
}

interface PlacementSetup {
  readonly A: Mat
  readonly Bt: Tensor
  readonly n: number
  readonly m: number
  readonly ps: Cpole[]
  readonly rankB: number
  readonly u0r: Mat
  readonly zr: Mat
  readonly kernels: { re: Mat; im: Mat; d: number }[]
  readonly X: Mat
  readonly method: 'YT' | 'KNV0'
  readonly rtol: number
  readonly maxIter: number
}

/** Everything before the sweeps: the poles, the bases of range(B) and its complement, the kernels and the initial X. */
function placementSetup(a: TensorLike, b: TensorLike, poles: readonly Pole[], options: PlaceOptions): PlacementSetup {
  const A = numbers2(a)
  const Bt = primalOf(asTensor(b))
  const n = asTensor(a).shape[0] ?? 0
  const m = Bt.shape[1] ?? 0
  const ps = orderPoles(poles)
  if (ps.length !== n) throw new MdmlError('value', `place: give ${n} poles, not ${ps.length}`)
  const rankB = matrixRank(Bt)
  for (const p of ps) {
    if (ps.filter((q) => q.re === p.re && q.im === p.im).length > rankB) {
      throw new MdmlError('value', 'place: a pole is repeated more than rank(B) times')
    }
  }
  const method = options.method ?? 'YT'
  const rtol = options.rtol ?? 1e-3
  const maxIter = options.maxIterations ?? 30
  // B = u0 z with u0 an orthonormal basis of range(B) and u1 of its complement.
  const { q, r } = qr(Bt)
  const u0 = Float64Array.from(primalOf(q).data)
  const u0cols = primalOf(q).shape[1] ?? m
  const z = Float64Array.from(primalOf(r).data)
  const u0r = new Float64Array(n * rankB)
  for (let i = 0; i < n; i++) for (let k = 0; k < rankB; k++) u0r[i * rankB + k] = u0[i * u0cols + k] ?? 0
  const zr = new Float64Array(rankB * m)
  for (let k = 0; k < rankB; k++) for (let j = 0; j < m; j++) zr[k * m + j] = z[k * m + j] ?? 0
  const kernels: { re: Mat; im: Mat; d: number }[] = []
  const X = new Float64Array(n * n)
  if (rankB === n) {
    for (let i = 0; i < n; i++) X[i * n + i] = 1
    return { A, Bt, n, m, ps, rankB, u0r, zr, kernels, X, method, rtol, maxIter }
  }
  const { basis: u1, cols: r1 } = complement(u0r, n, rankB)
  for (let j = 0; j < n; j++) {
    const p = ps[j] as Cpole
    const k = kernelFor(A, u1, n, r1, p)
    kernels.push(k)
    const sre = new Float64Array(n)
    const sim = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < k.d; c++) {
        sre[i] = (sre[i] ?? 0) + (k.re[i * k.d + c] ?? 0)
        sim[i] = (sim[i] ?? 0) + (k.im[i * k.d + c] ?? 0)
      }
    }
    const nr = Math.sqrt(norm(sre) ** 2 + norm(sim) ** 2)
    setCol(
      X,
      n,
      j,
      Float64Array.from(sre, (v) => v / nr),
    )
    if (!isReal(p)) {
      setCol(
        X,
        n,
        j + 1,
        Float64Array.from(sim, (v) => v / nr),
      )
      kernels.push(k)
      j += 1
    }
  }
  return { A, Bt, n, m, ps, rankB, u0r, zr, kernels, X, method, rtol, maxIter }
}

/** The gain, achieved poles and the rest of the result, from the setup and the eigenvector matrix the sweeps reached. */
function placementFinish(S: PlacementSetup, X: Mat, iterations: number, curRtol: number): PlacementResult {
  const { A, Bt, n, m, ps, rankB, u0r, zr } = S
  let gain: Tensor
  if (rankB === n) {
    const D = new Float64Array(n * n)
    for (let i = 0; i < n; i++) {
      const p = ps[i] as Cpole
      if (isReal(p)) D[i * n + i] = p.re
      else {
        D[i * n + i] = p.re
        D[(i + 1) * n + i + 1] = p.re
        D[i * n + i + 1] = -p.im
        D[(i + 1) * n + i] = p.im
        i += 1
      }
    }
    gain = primalOf(mul(lstsq(Bt, sub(tensor2(D, n, n), tensor2(A, n, n))), -1)) as Tensor
  } else {
    // M = X Lambda X^-1 with the 2 x 2 real blocks for complex pairs; K = z^-1 u0^T (M - A), then negated.
    const Xt = tensor2(X, n, n)
    const Mfull = primalOf(transpose(solve(transpose(Xt), transpose(matmul(Xt, blockFor(ps, n))))))
    const rhs = matmul(transpose(tensor2(u0r, n, rankB)), sub(Mfull, tensor2(A, n, n)))
    let k: Tensor
    try {
      k = primalOf(solve(tensor2(zr, rankB, m), rhs)) as Tensor
    } catch {
      throw new MdmlError('value', "place: these poles can't be placed (check controllability)")
    }
    gain = primalOf(mul(k, -1)) as Tensor
  }
  const cl = eigvals(sub(tensor2(A, n, n), matmul(Bt, gain)))
  const computed = orderPoles(Array.from(cl.re.data, (re, i) => ({ re, im: snap(cl.im.data[i] ?? 0) })))
  return { gain, requestedPoles: ps, computedPoles: computed, X: tensor2(X, n, n), iterations, rtol: curRtol }
}

/**
 * Pole placement (Kautsky, Nichols and Van Dooren, as scipy.signal.place_poles) as a stepped algorithm, one sweep per
 * step: the setup chooses each closed-loop eigenvector in the kernel of u1^T (A - lambda I), and each sweep updates the
 * eigenvectors of the real poles by KNV method 0 (one at a time) or Tits and Yang (pairwise, in scipy's order) so that
 * the eigenvector matrix becomes better conditioned, stopping when |det X| changes by less than `rtol`. It logs the
 * conditioning of X, the sweep and the change, and the closed-loop poles on request, and is done (`converged`) when the
 * change is small or no sweeping is needed (B of full rank, or a single input) and `completed` after `maxIterations`. The
 * state serialises; read the gain and poles with {@link placementOf}.
 *
 * @param a - A `[n, n]`.
 * @param b - B `[n, m]`.
 * @param poles - The desired poles.
 * @param options - Method, relative tolerance and sweep cap.
 * @returns The algorithm.
 * @throws MdmlError with code `value` for the wrong number of poles or a pole repeated more than rank(B) times.
 * @concept pole-placement
 * @example
 * ```ts
 * import { control } from 'mdml-engine/dynamics'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const alg = control.placeAlgorithm([[0, 1, 0], [0, 0, 1], [0, 0, 0]], [[0, 0], [1, 0], [0, 1]], [-1, -2, -3])
 * runAlgorithm(alg, { maxSteps: 30 }).state.iterations >= 1 // => true
 * ```
 */
export function placeAlgorithm(
  a: TensorLike,
  b: TensorLike,
  poles: readonly Pole[],
  options: PlaceOptions = {},
): Algorithm<PlacementState, PlacePath> {
  const S = placementSetup(a, b, poles, options)
  const { n, ps, rankB, kernels } = S
  const sweeps = rankB > 1 && rankB < n
  const order = S.method === 'KNV0' ? Array.from({ length: n }, (_, j) => [j, j] as [number, number]) : ytOrder(ps)
  return {
    name: 'dynamics/place-poles',
    config: { n, rankB, method: S.method, rtol: S.rtol },
    init: () => ({ X: Float64Array.from(S.X), iterations: 0, rtol: Number.NaN, stop: !sweeps }),
    step: (st, ctx) => {
      const X = Float64Array.from(st.X)
      const before = Math.abs(primalDet(X, n))
      for (const [i, j] of order) {
        const pi = ps[i] as Cpole
        const pj = ps[j] as Cpole
        if (!isReal(pi) || !isReal(pj)) continue
        const ki = kernels[i] as { re: Mat; d: number }
        const kj = kernels[j] as { re: Mat; d: number }
        if (i === j) knv0(X, n, ki.re, ki.d, i)
        else ytReal(X, n, ki.re, ki.d, kj.re, kj.d, i, j)
      }
      const after = Math.max(Math.sqrt(Number.EPSILON), Math.abs(primalDet(X, n)))
      const curRtol = Math.abs((after - before) / after)
      const stop = curRtol < S.rtol && after > Math.sqrt(Number.EPSILON)
      const sv = numbersOf(primalOf(svd(tensor2(X, n, n)).s))
      ctx.log('place/sweep', st.iterations + 1)
      ctx.log('place/change', curRtol)
      ctx.log('place/conditioning', (sv[0] ?? 0) / Math.max(sv[sv.length - 1] ?? 0, Number.MIN_VALUE))
      if (ctx.wants('place/poles')) {
        const res = placementFinish(S, X, st.iterations + 1, curRtol)
        ctx.log(
          'place/poles',
          res.computedPoles.flatMap((p) => [p.re, p.im]),
        )
      }
      return { X, iterations: st.iterations + 1, rtol: curRtol, stop }
    },
    done: (st) => (st.stop ? 'converged' : st.iterations >= S.maxIter ? 'completed' : undefined),
    serialise: (st) => ({
      X: numbersToJson(st.X),
      iterations: st.iterations,
      rtol: numberToJson(st.rtol),
      stop: st.stop,
    }),
    deserialise: (j) => {
      const o = j as unknown as { X: Json; iterations: number; rtol: Json; stop: boolean }
      return {
        X: Float64Array.from(numbersFromJson(o.X)),
        iterations: o.iterations,
        rtol: numberFromJson(o.rtol),
        stop: o.stop,
      }
    },
  }
}

/**
 * The gain and poles a pole-placement state gives: the closed loop is A - B K.
 *
 * @param a - A `[n, n]`.
 * @param b - B `[n, m]`.
 * @param poles - The desired poles the state was run with.
 * @param state - A state of {@link placeAlgorithm}.
 * @param options - The options it was run with.
 * @returns The gain, requested and achieved poles, eigenvector matrix and sweep count.
 * @throws MdmlError with code `value` when the poles cannot be placed.
 * @concept pole-placement
 * @example
 * ```ts
 * import { control } from 'mdml-engine/dynamics'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const a = [[0, 1], [0, 0]]
 * const b = [[0], [1]]
 * control.placementOf(a, b, [-1, -2], runAlgorithm(control.placeAlgorithm(a, b, [-1, -2]), { maxSteps: 5 }).state).iterations // => 0
 * ```
 */
export function placementOf(
  a: TensorLike,
  b: TensorLike,
  poles: readonly Pole[],
  state: PlacementState,
  options: PlaceOptions = {},
): PlacementResult {
  return placementFinish(placementSetup(a, b, poles, options), state.X, state.iterations, state.rtol)
}

/**
 * State-feedback pole placement after Kautsky, Nichols and Van Dooren (scipy.signal.place_poles): the closed-loop
 * eigenvectors are chosen in the kernels of u1^T (A - lambda I) and improved for robustness by KNV method 0 sweeps
 * or Tits and Yang's pairwise updates (the default), then K follows from B K = A - X Lambda X^-1 in the range of B.
 * With as many independent inputs as states the gain is the least-squares solution. Complex poles keep their
 * initial eigenvectors (the iterations improve the real ones).
 *
 * @param a - A `[n, n]`.
 * @param b - B `[n, m]`.
 * @param poles - The desired poles (complex ones as {re, im} with their conjugates).
 * @param options - Method (`YT` or `KNV0`), relative tolerance (default 1e-3) and iteration cap (default 30).
 * @returns The gain, requested and achieved poles, eigenvector matrix and iteration count.
 * @throws MdmlError with code `value` for the wrong number of poles, poles repeated more than rank(B) times, or
 * poles that cannot be placed. {@link placeAlgorithm} run to the end.
 * @concept pole-placement
 * @example
 * ```ts
 * import { control } from 'mdml-engine/dynamics'
 * import { toArray } from 'mdml-engine/tensor'
 * // The double integrator with poles -1, -2: K = [2, 3].
 * toArray(control.place([[0, 1], [0, 0]], [[0], [1]], [-1, -2]).gain).map((r) => r.map((v) => Math.round(v * 1e10) / 1e10)) // => [[2, 3]]
 * ```
 */
export function place(
  a: TensorLike,
  b: TensorLike,
  poles: readonly Pole[],
  options: PlaceOptions = {},
): PlacementResult {
  const run = runAlgorithm(placeAlgorithm(a, b, poles, options), { maxSteps: options.maxIterations ?? 30 })
  return placementOf(a, b, poles, run.state, options)
}

function snap(v: number): number {
  return Math.abs(v) < 1e-12 ? 0 : v
}

function blockFor(ps: readonly Cpole[], n: number): Tensor {
  // Lambda^T blocks so that A_cl [x_r, x_i] = [x_r, x_i] [[a, b], [-b, a]].
  const D = new Float64Array(n * n)
  for (let i = 0; i < n; i++) {
    const p = ps[i] as Cpole
    if (isReal(p)) D[i * n + i] = p.re
    else {
      D[i * n + i] = p.re
      D[(i + 1) * n + i + 1] = p.re
      D[i * n + i + 1] = p.im
      D[(i + 1) * n + i] = -p.im
      i += 1
    }
  }
  return tensor2(D, n, n)
}

function primalDet(X: Mat, n: number): number {
  return numbersOf(primalOf(det(tensor2(X, n, n))))[0] ?? 0
}

/**
 * Ackermann's formula for single-input pole placement (python-control's `acker`): K = [0 ... 0 1] C^-1 phi(A), with
 * C the controllability matrix and phi the desired characteristic polynomial. Exact but poorly conditioned for large
 * n; prefer {@link place}.
 *
 * @param a - A `[n, n]`.
 * @param b - B `[n, 1]`.
 * @param poles - The desired poles.
 * @returns K `[1, n]`.
 * @throws MdmlError with code `value` for more than one input or the wrong number of poles.
 * @concept pole-placement
 * @example
 * ```ts
 * import { control } from 'mdml-engine/dynamics'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(control.acker([[0, 1], [0, 0]], [[0], [1]], [-1, -2])) // => [[2, 3]]
 * ```
 */
export function acker(a: TensorLike, b: TensorLike, poles: readonly Pole[]): Tensor {
  const A = asTensor(a)
  const B = asTensor(b)
  const n = A.shape[0] ?? 0
  if ((B.shape[1] ?? 1) !== 1) throw new MdmlError('value', 'acker: one input only')
  const ps = orderPoles(poles)
  if (ps.length !== n) throw new MdmlError('value', `acker: give ${n} poles`)
  // Characteristic polynomial coefficients (highest power first) from the roots, in real arithmetic by pairs.
  let coeffs = [1]
  const mulPoly = (c: number[], f: number[]): number[] => {
    const out = new Array<number>(c.length + f.length - 1).fill(0)
    c.forEach((x, i) => {
      f.forEach((y, j) => {
        out[i + j] = (out[i + j] ?? 0) + x * y
      })
    })
    return out
  }
  for (let i = 0; i < n; i++) {
    const p = ps[i] as Cpole
    if (isReal(p)) coeffs = mulPoly(coeffs, [1, -p.re])
    else {
      coeffs = mulPoly(coeffs, [1, -2 * p.re, p.re * p.re + p.im * p.im])
      i += 1
    }
  }
  // phi(A) = sum c_k A^(n-k)
  let phi = mul(eye(n), 0)
  coeffs.forEach((c, k) => {
    phi = add(phi, mul(matrixPower(A, n - k), c))
  })
  const C = ctrb(A, B)
  const en = fromNumbers(
    Float64Array.from({ length: n }, (_, i) => (i === n - 1 ? 1 : 0)),
    [1, n],
  )
  return primalOf(matmul(en, solve(C, phi))) as Tensor
}
