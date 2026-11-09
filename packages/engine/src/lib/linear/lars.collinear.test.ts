import { describe, expect, it } from 'vitest'
import { lassoCoordinateDescent } from '../optim/sparse.ts'
import { key, normal } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { coefficientsAt, larsPath } from './lars.ts'

type Data = { readonly x: number[][]; readonly y: number[] }

/** Unit-norm random columns, `n` long. */
function columns(n: number, k: number, seed: number): number[][] {
  const g = toFlat(normal(key(seed), [n * k]))
  return Array.from({ length: k }, (_, j) => {
    const col = Array.from({ length: n }, (_, i) => g[i * k + j] ?? 0)
    const norm = Math.hypot(...col)
    return col.map((v) => v / norm)
  })
}

function fromColumns(cols: number[][], y: number[]): Data {
  return { x: (cols[0] ?? []).map((_, i) => cols.map((c) => c[i] ?? 0)), y }
}

/** A sparse truth over the first columns plus noise. */
function response(cols: number[][], seed: number, noise: number): number[] {
  const e = toFlat(normal(key(seed + 999), [cols[0]?.length ?? 0]))
  return (cols[0] ?? []).map((_, i) => 3 * (cols[0]?.[i] ?? 0) - 2 * (cols[1]?.[i] ?? 0) + noise * (e[i] ?? 0))
}

/** Generic columns (no repeats): the lasso path has variables leaving and re-entering. */
function generic(n: number, d: number, seed: number): Data {
  const g = toFlat(normal(key(seed + 5000), [n * (d + 1)]))
  const x = Array.from({ length: n }, (_, i) =>
    Array.from({ length: d }, (_, j) => (g[i * (d + 1) + j] ?? 0) + 0.7 * (j > 0 ? (g[i * (d + 1) + j - 1] ?? 0) : 0)),
  )
  return { x, y: x.map((r, i) => (g[i * (d + 1) + d] ?? 0) + 2 * (r[0] ?? 0)) }
}

/** Atoms with exact repeats, negated repeats, rescaled repeats and repeats to 1e-9. */
function degenerate(n: number, k: number, seed: number, kind: 'exact' | 'near'): Data {
  const cols = columns(n, k, seed)
  const extra: number[][] = []
  for (let m = 0; m < 1 + (seed % 5); m++) {
    const src = cols[(seed * 7 + m * 3) % k] ?? []
    const sign = m % 2 === 0 ? 1 : -1
    const scale = m % 3 === 2 ? 2.5 : 1
    const bump = (v: number, i: number): number =>
      sign * scale * v + (kind === 'near' ? 1e-9 * Math.sin(i * (m + 1)) : 0)
    extra.push(src.map(bump))
  }
  const all = [...cols, ...extra]
  return fromColumns(all, response(cols, seed, 0.3))
}

/** The largest violation of the lasso optimality conditions of `coef` at penalty `alpha` (the penalty is alpha * n). */
function kkt({ x, y }: Data, coef: number[], alpha: number, fitIntercept: boolean, positive = false): number {
  const n = x.length
  const d = x[0]?.length ?? 0
  const xm = Array.from({ length: d }, (_, j) => (fitIntercept ? x.reduce((a, r) => a + (r[j] ?? 0), 0) / n : 0))
  const ym = fitIntercept ? y.reduce((a, b) => a + b, 0) / n : 0
  const r = y.map((yi, i) => yi - ym - coef.reduce((a, c, j) => a + c * ((x[i]?.[j] ?? 0) - (xm[j] ?? 0)), 0))
  let worst = 0
  for (let j = 0; j < d; j++) {
    const c = r.reduce((a, ri, i) => a + ((x[i]?.[j] ?? 0) - (xm[j] ?? 0)) * ri, 0) / n
    const cj = coef[j] ?? 0
    if (positive) worst = Math.max(worst, -cj, cj === 0 ? Math.max(0, c - alpha) : Math.abs(c - alpha))
    else worst = Math.max(worst, cj === 0 ? Math.max(0, Math.abs(c) - alpha) : Math.abs(c - alpha * Math.sign(cj)))
  }
  return worst
}

/** Checks the optimality conditions at fractions of the top penalty, where the path reaches. */
function expectOptimal(data: Data, fitIntercept: boolean, positive = false): void {
  const path = larsPath(data, { method: 'lasso', fitIntercept, positive })
  const top = path.alphas[0] ?? 1
  const last = path.alphas[path.alphas.length - 1] ?? 0
  for (const f of [0.6, 0.3, 0.1, 0.03]) {
    const alpha = top * f
    if (alpha < last) continue
    expect(kkt(data, coefficientsAt(path, alpha), alpha, fitIntercept, positive), `alpha ${alpha}`).toBeLessThan(
      1e-6 * top,
    )
  }
}

describe('larsPath stays on the lasso optimum', () => {
  it('when variables leave and re-enter the active set (no zero-length knot after a drop)', () => {
    for (const seed of [4, 8, 13, 29, 34, 38, 65])
      for (const fitIntercept of [true, false])
        expectOptimal(generic(8 + (seed % 20), 3 + (seed % 9), seed), fitIntercept)
  })

  it('on designs with repeated, negated, rescaled and nearly repeated columns, wide or tall', () => {
    for (let seed = 0; seed < 60; seed++) {
      const n = 6 + (seed % 10)
      const k = 5 + (seed % 17)
      for (const kind of ['exact', 'near'] as const)
        for (const fitIntercept of [true, false]) expectOptimal(degenerate(n, k, seed, kind), fitIntercept)
    }
  })

  it('with non-negative coefficients, on generic and degenerate designs', () => {
    for (let seed = 0; seed < 40; seed++) {
      const n = 8 + (seed % 12)
      expectOptimal(generic(n, 3 + (seed % 7), seed), seed % 2 === 0, true)
      expectOptimal(degenerate(n, 5 + (seed % 9), seed, seed % 4 < 2 ? 'exact' : 'near'), seed % 2 === 0, true)
    }
  })

  it('refuses non-negative coefficients for the plain path', () => {
    expect(() => larsPath(generic(10, 3, 1), { method: 'lar', positive: true })).toThrow(/positive/)
  })

  it('matches coordinate descent in fitted values on a design with a repeated column', () => {
    const cols = columns(30, 6, 3)
    const data = fromColumns([...cols, cols[0] ?? [], cols[2] ?? []], response(cols, 3, 0.1))
    const path = larsPath(data, { method: 'lasso', fitIntercept: false })
    const alpha = (path.alphas[0] ?? 1) * 0.2
    const n = data.x.length
    const cd = toFlat(lassoCoordinateDescent(data.x, data.y, alpha * n, { tol: 1e-14, maxIterations: 100000 }).coef)
    const lars = coefficientsAt(path, alpha)
    for (const row of data.x) {
      const a = row.reduce((s, v, j) => s + v * (cd[j] ?? 0), 0)
      const b = row.reduce((s, v, j) => s + v * (lars[j] ?? 0), 0)
      expect(Math.abs(a - b)).toBeLessThan(1e-5)
    }
  })
})
