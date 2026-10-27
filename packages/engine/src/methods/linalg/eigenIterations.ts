/**
 * Eigenvalue iterations watched, as methods of their configuration alone: a symmetric matrix with chosen eigenvalues
 * and a random eigenbasis from the key, and either the power iteration (a unit vector turning towards the leading
 * eigenvector while its Rayleigh quotient converges) or the unshifted QR iteration (the matrix itself converging to a
 * diagonal of eigenvalues as its off-diagonal mass drains), one iteration per step. The unit circle in the leading
 * eigenplane and its image under the matrix, with the current iterate, are a probe.
 */

import { type CurveSet, curveSet } from '../../lib/exposition/index.ts'
import { qr } from '../../lib/numerics/linalg/index.ts'
import { type Key, normal } from '../../lib/random/index.ts'
import { dict, int, real } from '../../lib/space/index.ts'
import { matmul, tensor, toFlat, transpose } from '../../lib/tensor/index.ts'
import {
  type Algorithm,
  type Config,
  type Contract,
  defineMethod,
  derived,
  Log,
  type Method,
  scalar,
  tensor as tensorSource,
} from '../../lib/trace/index.ts'

/** The state of an eigen iteration: the matrix in play, the iterate (the vector or the diagonal) and its estimate. */
export interface EigenState {
  readonly k: number
  /** The matrix: constant under the power iteration, the current A_k under QR. */
  readonly a: readonly (readonly number[])[]
  /** The unit vector (power) or the diagonal of A_k (QR). */
  readonly v: readonly number[]
}

/** The matrix a configuration and key build, with its eigenvalues (ascending) and eigenvectors (as columns). */
interface Problem {
  readonly n: number
  readonly a: number[][]
  readonly eigenvalues: number[]
  readonly eigenvectors: number[][]
}

const PROBLEMS = new Map<string, Problem>()
function problemOf(c: Config, key: Key): Problem {
  const id = `${JSON.stringify(c)}|${key.words.join(',')}`
  const cached = PROBLEMS.get(id)
  if (cached !== undefined) return cached
  const n = c['n'] as number
  const spread = c['spread'] as number
  // Eigenvalues spaced geometrically from 1 to the spread, so the convergence rate is their ratio.
  const eigenvalues = Array.from({ length: n }, (_, i) => spread ** (n === 1 ? 0 : i / (n - 1)))
  const q = qr(normal(key, [n, n])).q
  const a = matmul(
    matmul(q, tensor(eigenvalues.map((l, i) => eigenvalues.map((_, j) => (i === j ? l : 0))))),
    transpose(q),
  )
  const rows = (flat: number[]): number[][] => Array.from({ length: n }, (_, i) => flat.slice(i * n, (i + 1) * n))
  const problem = { n, a: rows(toFlat(a)), eigenvalues, eigenvectors: rows(toFlat(q)) }
  PROBLEMS.set(id, problem)
  return problem
}

const CONFIG = dict({
  n: int(2, 8, { default: 4, label: 'size' }),
  spread: real(1.5, 20, { default: 4, label: 'largest / smallest eigenvalue', log: true }),
})

const matvec = (a: readonly (readonly number[])[], v: readonly number[]): number[] =>
  a.map((row) => row.reduce((acc, x, j) => acc + x * (v[j] ?? 0), 0))
const norm = (v: readonly number[]): number => Math.sqrt(v.reduce((acc, x) => acc + x * x, 0))

// The unit circle in the plane of the two leading eigenvectors, its image under A, and the iterate's projection.
function actionRecord(p: Problem, v: readonly number[] | undefined): CurveSet {
  const n = p.n
  const l1 = p.eigenvalues[n - 1] ?? 1
  const l2 = p.eigenvalues[n - 2] ?? 1
  const q1 = p.eigenvectors.map((row) => row[n - 1] ?? 0)
  const q2 = p.eigenvectors.map((row) => row[n - 2] ?? 0)
  const thetas = Array.from({ length: 73 }, (_, i) => (2 * Math.PI * i) / 72)
  const series: { name: string; x: number[]; y: number[] }[] = [
    { name: 'unit circle', x: thetas.map(Math.cos), y: thetas.map(Math.sin) },
    { name: 'its image under A', x: thetas.map((t) => l1 * Math.cos(t)), y: thetas.map((t) => l2 * Math.sin(t)) },
  ]
  if (v !== undefined) {
    const dot = (u: readonly number[]): number => u.reduce((acc, x, i) => acc + x * (v[i] ?? 0), 0)
    const c1 = dot(q1)
    const c2 = dot(q2)
    series.push({ name: 'v', x: [0, c1], y: [0, c2] }, { name: 'A v', x: [0, l1 * c1], y: [0, l2 * c2] })
  }
  return curveSet({ series, xLabel: 'along the leading eigenvector', yLabel: 'along the second' })
}

function contractOf(kind: 'power' | 'qr', c: Config, key: Key): Contract<EigenState> {
  const p = problemOf(c, key)
  return {
    'iteration/matrix': tensorSource({
      role: 'state',
      shape: [p.n, p.n],
      description: kind === 'qr' ? 'A_k, converging to a diagonal' : 'the matrix',
    }),
    'iteration/vector': tensorSource({
      role: 'state',
      shape: [p.n],
      description: kind === 'qr' ? 'the diagonal of A_k: the eigenvalue estimates' : 'the unit iterate',
    }),
    'iteration/rayleigh': scalar({
      description: 'the Rayleigh quotient of the iterate (the leading eigenvalue estimate)',
    }),
    'iteration/error': scalar({
      description: 'the largest error of the eigenvalue estimates against the true eigenvalues',
      direction: 'lower',
    }),
    'iteration/offDiagonal': scalar({
      description: 'the Frobenius norm of the off-diagonal part of A_k',
      direction: 'lower',
    }),
    'matrix/action': derived<EigenState>({
      description: 'the unit circle of the leading eigenplane, its image under A, and the iterate',
      compute: (s) => actionRecord(p, kind === 'power' ? s.v : undefined),
    }),
  }
}

function makeOf(kind: 'power' | 'qr', c: Config, key: Key): Algorithm<EigenState> {
  const p = problemOf(c, key)
  const top = p.eigenvalues[p.n - 1] ?? 1
  const sorted = [...p.eigenvalues].sort((x, y) => x - y)
  return {
    name: `eigen-${kind}`,
    config: { n: p.n, spread: c['spread'] as number },
    init: (k) => {
      if (kind === 'qr') return { k: 0, a: p.a, v: p.a.map((row, i) => row[i] ?? 0) }
      const v0 = Array.from(normal(k, [p.n]).data)
      const s = norm(v0)
      return { k: 0, a: p.a, v: v0.map((x) => x / s) }
    },
    step: (state, ctx) => {
      if (kind === 'power') {
        const w = matvec(state.a, state.v)
        const s = norm(w)
        const v = w.map((x) => x / s)
        const rayleigh = v.reduce((acc, x, i) => acc + x * (matvec(state.a, v)[i] ?? 0), 0)
        ctx.log('iteration/matrix', state.a)
        ctx.log('iteration/vector', v)
        ctx.log('iteration/rayleigh', rayleigh)
        ctx.log('iteration/error', Math.abs(rayleigh - top))
        return { k: state.k + 1, a: state.a, v }
      }
      const { q, r } = qr(tensor(state.a))
      const flat = toFlat(matmul(r, q))
      const a = Array.from({ length: p.n }, (_, i) => flat.slice(i * p.n, (i + 1) * p.n))
      const diagonal = a.map((row, i) => row[i] ?? 0)
      const off = Math.sqrt(a.reduce((acc, row, i) => acc + row.reduce((s, x, j) => (i === j ? s : s + x * x), 0), 0))
      const estimates = [...diagonal].sort((x, y) => x - y)
      const error = Math.max(...estimates.map((e, i) => Math.abs(e - (sorted[i] ?? 0))))
      ctx.log('iteration/matrix', a)
      ctx.log('iteration/vector', diagonal)
      ctx.log('iteration/offDiagonal', off)
      ctx.log('iteration/error', error)
      return { k: state.k + 1, a, v: diagonal }
    },
    serialise: (s) => ({ k: s.k, a: s.a.map((row) => [...row]), v: [...s.v] }),
    deserialise: (j) => j as unknown as EigenState,
  }
}

/**
 * The power iteration on a symmetric matrix: v ← A v / ‖A v‖, one iteration per step.
 *
 * Configuration: `n` (the size) and `spread` (the ratio of the largest to the smallest eigenvalue; the eigenvalues
 * are spaced geometrically between them, so the second-to-first ratio sets the convergence rate). Paths:
 * `iteration/vector` (the unit iterate, full), `iteration/rayleigh`, `iteration/error`, `iteration/matrix`, and the
 * probe `matrix/action` (the leading eigenplane's unit circle, its image under A, and the iterate).
 */
export const eigenPower: Method<EigenState, Config> = defineMethod<EigenState, Config>({
  name: 'linalg/power-iteration',
  config: CONFIG,
  make: (c, key) => makeOf('power', c, key),
  contract: (c, key) => contractOf('power', c, key),
  defaults: {
    'iteration/matrix': Log({ detail: 'full' }),
    'iteration/vector': Log({ detail: 'full' }),
    'iteration/rayleigh': Log(),
    'iteration/error': Log(),
  },
})

/**
 * The unshifted QR iteration on a symmetric matrix: A_{k+1} = R_k Q_k where A_k = Q_k R_k, one iteration per step.
 *
 * Configuration as {@link eigenPower}. Paths: `iteration/matrix` (A_k, full), `iteration/vector` (its diagonal, the
 * eigenvalue estimates, full), `iteration/offDiagonal`, `iteration/error`, and the probe `matrix/action`.
 */
export const eigenQr: Method<EigenState, Config> = defineMethod<EigenState, Config>({
  name: 'linalg/qr-iteration',
  config: CONFIG,
  make: (c, key) => makeOf('qr', c, key),
  contract: (c, key) => contractOf('qr', c, key),
  defaults: {
    'iteration/matrix': Log({ detail: 'full' }),
    'iteration/vector': Log({ detail: 'full' }),
    'iteration/offDiagonal': Log(),
    'iteration/error': Log(),
  },
})
