/**
 * Symbolic regression on the edges of a Kolmogorov-Arnold network: what makes a KAN more than a small network is that
 * each edge is a one-dimensional function you can look at, and often name. `suggestSymbolic` fits every function of a
 * table (`symbolicFunctions`: `x`, `x^2`, `sin`, `exp`, `log`, ...) to an edge's activation on the data in the affine form
 * `a f(b x + c) + d` (a grid over `b` and `c` with `a` and `d` solved by regression, then Nelder-Mead on `b` and `c`) and
 * ranks them by `R^2` less a small price on complexity, as pykan's `suggest_symbolic` does. `fixSymbolic` fixes every edge
 * to its best function, and `symbolicFormula` composes the fixed edges through the nodes into one closed-form expression
 * per output, a `symbolic` expression that can be printed, differentiated or compiled.
 */

import { OmaiError } from '../../base/index.ts'
import { nelderMead } from '../../optim/index.ts'
import {
  abs as absExpr,
  add as addExpr,
  type Expr,
  exp as expExpr,
  fn,
  log as logExpr,
  mul as mulExpr,
  num,
  pow as powExpr,
  sqrt as sqrtExpr,
  sym,
  toText,
} from '../../symbolic/index.ts'
import { asTensor, type TensorLike, toFlat, type Value } from '../../tensor/index.ts'
import type { KAN, KanState } from './kan.ts'

/** One function of the table: its values, how it is written as an expression, and how complicated it is. */
export interface SymbolicFunction {
  readonly name: string
  readonly f: (x: number) => number
  readonly expr: (arg: Expr) => Expr
  /** A rough count of how involved the function is (pykan's complexity, used to break ties towards simple ones). */
  readonly complexity: number
}

const entry = (
  name: string,
  complexity: number,
  f: (x: number) => number,
  expr: (arg: Expr) => Expr,
): SymbolicFunction => ({ name, f, expr, complexity })

/**
 * The functions the symbolic fit chooses among, by name: powers and their reciprocals, roots, the exponential and
 * logarithm, the trigonometric and hyperbolic functions, `abs`, `arctan`, the Gaussian and zero. Each is fitted as
 * `a f(b x + c) + d`; with the phase `c`, `cos` and `sin` are one family, so `cos` is priced above `sin` and a sinusoid is
 * reported as `sin`.
 *
 * @example
 * ```ts
 * import { symbolicFunctions } from 'omai-engine/nn'
 * symbolicFunctions['x^2']?.f(3) // => 9
 * Object.keys(symbolicFunctions).includes('sin') // => true
 * ```
 */
export const symbolicFunctions: Readonly<Record<string, SymbolicFunction>> = Object.fromEntries(
  [
    entry(
      'x',
      1,
      (x) => x,
      (z) => z,
    ),
    entry(
      'x^2',
      2,
      (x) => x * x,
      (z) => powExpr(z, 2),
    ),
    entry(
      'x^3',
      3,
      (x) => x ** 3,
      (z) => powExpr(z, 3),
    ),
    entry(
      'x^4',
      3,
      (x) => x ** 4,
      (z) => powExpr(z, 4),
    ),
    entry(
      'x^5',
      3,
      (x) => x ** 5,
      (z) => powExpr(z, 5),
    ),
    entry(
      '1/x',
      2,
      (x) => 1 / x,
      (z) => powExpr(z, -1),
    ),
    entry(
      '1/x^2',
      2,
      (x) => 1 / (x * x),
      (z) => powExpr(z, -2),
    ),
    entry(
      '1/x^3',
      3,
      (x) => 1 / x ** 3,
      (z) => powExpr(z, -3),
    ),
    entry('sqrt', 2, Math.sqrt, (z) => sqrtExpr(z)),
    entry(
      '1/sqrt',
      2,
      (x) => 1 / Math.sqrt(x),
      (z) => powExpr(sqrtExpr(z), -1),
    ),
    entry('exp', 2, Math.exp, (z) => expExpr(z)),
    entry('log', 2, Math.log, (z) => logExpr(z)),
    entry('abs', 3, Math.abs, (z) => absExpr(z)),
    entry('sin', 2, Math.sin, (z) => fn('sin', z)),
    entry('cos', 3, Math.cos, (z) => fn('cos', z)),
    entry('tan', 3, Math.tan, (z) => fn('tan', z)),
    entry('tanh', 3, Math.tanh, (z) => fn('tanh', z)),
    entry('arctan', 3, Math.atan, (z) => fn('atan', z)),
    entry(
      'gaussian',
      3,
      (x) => Math.exp(-x * x),
      (z) => expExpr(mulExpr(-1, powExpr(z, 2))),
    ),
    entry(
      '0',
      0,
      () => 0,
      () => num(0),
    ),
  ].map((e) => [e.name, e]),
)

/** An edge fitted by a function of the table: `a f(b x + c) + d`, and how well it fits. */
export interface SymbolicFit {
  readonly name: string
  readonly a: number
  readonly b: number
  readonly c: number
  readonly d: number
  /** The coefficient of determination of the fit on the points it was made on. */
  readonly r2: number
  /** The ranking score: `r2` less the complexity price. */
  readonly score: number
}

/** Options of the symbolic fit. */
export interface SymbolicOptions {
  /** The names of the table to try (default all of `symbolicFunctions`). */
  readonly library?: readonly string[]
  /** What each unit of complexity costs in the ranking score (default 1e-3). */
  readonly complexityPenalty?: number
}

const B_GRID = [0.3, 0.6, 1, 1.6, 2.5, 4, 6.5, 10]
const C_GRID = [-2, -1, -0.5, 0, 0.5, 1, 2, ...Array.from({ length: 8 }, (_, k) => ((k - 3.5) * Math.PI) / 4)]

/** The affine-in-output fit for one function at a given inner `(b, c)`: `a`, `d` and the squared error. */
function affine(
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  f: (x: number) => number,
  b: number,
  c: number,
): { a: number; d: number; sse: number } | undefined {
  const n = xs.length
  const z = new Float64Array(n)
  let mz = 0
  let my = 0
  for (let k = 0; k < n; k++) {
    const v = f(b * (xs[k] as number) + c)
    if (!Number.isFinite(v) || Math.abs(v) > 1e12) return undefined
    z[k] = v
    mz += v
    my += ys[k] as number
  }
  mz /= n
  my /= n
  let szz = 0
  let szy = 0
  let syy = 0
  for (let k = 0; k < n; k++) {
    const dz = (z[k] as number) - mz
    const dy = (ys[k] as number) - my
    szz += dz * dz
    szy += dz * dy
    syy += dy * dy
  }
  const a = szz > 1e-24 ? szy / szz : 0
  return { a, d: my - a * mz, sse: Math.max(0, syy - a * szy) }
}

/**
 * Fit one function of the table to points in the form `a f(b x + c) + d`.
 *
 * @param xs - The inputs.
 * @param ys - The edge's values at them.
 * @param f - The function.
 * @returns The fit (`undefined` when the function is undefined at all of the starting points).
 */
function fitOne(
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  f: SymbolicFunction,
): Omit<SymbolicFit, 'score'> | undefined {
  const n = xs.length
  let my = 0
  for (let k = 0; k < n; k++) my += ys[k] as number
  my /= n
  let sst = 0
  for (let k = 0; k < n; k++) sst += ((ys[k] as number) - my) ** 2
  if (f.name === '0') return { name: '0', a: 0, b: 1, c: 0, d: my, r2: sst < 1e-20 && Math.abs(my) < 1e-6 ? 1 : 0 }
  if (sst < 1e-20) return { name: f.name, a: 0, b: 1, c: 0, d: my, r2: 1 }
  let lo = Number.POSITIVE_INFINITY
  let hi = Number.NEGATIVE_INFINITY
  for (let k = 0; k < n; k++) {
    lo = Math.min(lo, xs[k] as number)
    hi = Math.max(hi, xs[k] as number)
  }
  const starts: { b: number; c: number; sse: number }[] = []
  const bs = [...B_GRID, ...B_GRID.map((v) => -v)]
  for (const b of bs) {
    // The shifts that put `b x + c` at a fixed distance above zero on the data (for log, sqrt and the reciprocals).
    const positive = [0.1, 0.5, 1, 2].map((delta) => delta - b * (b > 0 ? lo : hi))
    for (const c of [...C_GRID, ...positive]) {
      const fit = affine(xs, ys, f.f, b, c)
      if (fit !== undefined) starts.push({ b, c, sse: fit.sse })
    }
  }
  if (starts.length === 0) return undefined
  starts.sort((p, q) => p.sse - q.sse)
  let best = starts[0] as { b: number; c: number; sse: number }
  for (const s of starts.slice(0, 3)) {
    const cost = (v: Value): Value => {
      const [b, c] = toFlat(v) as unknown as [number, number]
      const ok = Math.abs(b) >= 0.05 && Math.abs(b) <= 60
      const fit = ok ? affine(xs, ys, f.f, b, c) : undefined
      return asTensor(fit === undefined ? 1e30 : fit.sse)
    }
    const r = nelderMead(cost, asTensor([s.b, s.c]), { xatol: 1e-9, fatol: 1e-14, maxEvaluations: 400 })
    const [b, c] = toFlat(r.x as Value) as unknown as [number, number]
    const fit = affine(xs, ys, f.f, b, c)
    if (fit !== undefined && fit.sse < best.sse) best = { b, c, sse: fit.sse }
  }
  const fit = affine(xs, ys, f.f, best.b, best.c) as { a: number; d: number; sse: number }
  return { name: f.name, a: fit.a, b: best.b, c: best.c, d: fit.d, r2: 1 - fit.sse / sst }
}

/**
 * Rank the functions of the table by how well `a f(b x + c) + d` fits the points `(xs, ys)`: `R^2` less
 * `complexityPenalty` times the complexity, best first.
 *
 * @param xs - The inputs.
 * @param ys - The values (an edge's activations).
 * @param options - The functions to try and the complexity price.
 * @returns The fits, best first (functions undefined on the data are left out).
 * @concept symbolic-regression
 * @example
 * ```ts
 * import { rankSymbolic } from 'omai-engine/nn'
 * const xs = Array.from({ length: 40 }, (_, i) => -1 + i / 19.5)
 * rankSymbolic(xs, xs.map((x) => 2 * x * x - 1))[0]?.name // => 'x^2'
 * ```
 */
export function rankSymbolic(
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  options: SymbolicOptions = {},
): SymbolicFit[] {
  if (xs.length !== ys.length || xs.length < 3)
    throw new OmaiError('shape', 'rankSymbolic: need matching xs and ys, at least 3')
  const price = options.complexityPenalty ?? 1e-3
  const names = options.library ?? Object.keys(symbolicFunctions)
  const fits: SymbolicFit[] = []
  for (const name of names) {
    const f = symbolicFunctions[name]
    if (f === undefined) throw new OmaiError('value', `rankSymbolic: unknown function ${name}`)
    const fit = fitOne(xs, ys, f)
    if (fit !== undefined) fits.push({ ...fit, score: fit.r2 - price * f.complexity })
  }
  return fits.sort((p, q) => q.score - p.score)
}

/**
 * Suggest the symbolic function of an edge of a trained network: the edge's activations on the data, ranked
 * (pykan's `suggest_symbolic`).
 *
 * @param kan - The network.
 * @param state - Its parameters and buffers.
 * @param x - Inputs `[n, n_in]`.
 * @param layer - The layer of the edge.
 * @param i - The source node.
 * @param o - The target node.
 * @param options - The functions to try and the complexity price.
 * @returns The fits, best first.
 * @concept symbolic-regression
 * @example
 * ```ts
 * import { KAN, suggestSymbolic } from 'omai-engine/nn'
 * import { key } from 'omai-engine/random'
 * const net = new KAN([1, 1], { key: key(0) })
 * const x = Array.from({ length: 20 }, (_, i) => [-1 + i / 9.5])
 * suggestSymbolic(net, net.state, x, 0, 0, 0).length > 0 // => true
 * ```
 */
export function suggestSymbolic(
  kan: KAN,
  state: KanState,
  x: TensorLike,
  layer: number,
  i: number,
  o: number,
  options: SymbolicOptions = {},
): SymbolicFit[] {
  const t = kan.trace(state.params, x, state.buffers)
  const input = t.inputs[layer]
  const acts = t.activations[layer]
  if (input === undefined || acts === undefined) throw new OmaiError('value', `suggestSymbolic: no layer ${layer}`)
  const [n, din, dout] = [acts.shape[0] ?? 0, acts.shape[1] ?? 0, acts.shape[2] ?? 0]
  const h = toFlat(input)
  const a = toFlat(acts)
  const xs = Float64Array.from({ length: n }, (_, r) => h[r * din + i] as number)
  const ys = Float64Array.from({ length: n }, (_, r) => a[(r * din + i) * dout + o] as number)
  return rankSymbolic(xs, ys, options)
}

/** An edge fixed to a function: `null` for an edge that was switched off (it contributes nothing). */
export type SymbolicEdge = Omit<SymbolicFit, 'score'> | null

/** A network whose edges are fixed to symbolic functions: `layers[l].edges[o][i]` is the edge from node `i` to node `o`. */
export interface SymbolicKan {
  readonly widths: readonly number[]
  readonly layers: readonly {
    readonly edges: readonly (readonly (SymbolicEdge | undefined)[])[]
    readonly bias: readonly number[]
  }[]
}

/** Options of {@link fixSymbolic}. */
export interface FixSymbolicOptions extends SymbolicOptions {
  /** Leave an edge unfixed when its best fit has `R^2` below this (default 0.9). */
  readonly r2Threshold?: number
}

/**
 * Fix every edge of a trained network to its best symbolic function on the data (the best `suggestSymbolic` fit, or
 * the zero function for a switched-off edge); edges whose best `R^2` is below the threshold are left unfixed
 * (`undefined`).
 *
 * @param kan - The network.
 * @param state - Its parameters and buffers.
 * @param x - Inputs `[n, n_in]`.
 * @param options - The functions, the complexity price and the `R^2` threshold.
 * @returns The symbolic network.
 * @concept symbolic-regression
 * @example
 * ```ts
 * import { KAN, fixSymbolic } from 'omai-engine/nn'
 * import { key } from 'omai-engine/random'
 * const net = new KAN([1, 1], { key: key(0) })
 * const x = Array.from({ length: 20 }, (_, i) => [-1 + i / 9.5])
 * fixSymbolic(net, net.state, x, { r2Threshold: 2 }).layers[0]?.edges[0]?.[0] // => undefined
 * ```
 */
export function fixSymbolic(kan: KAN, state: KanState, x: TensorLike, options: FixSymbolicOptions = {}): SymbolicKan {
  const threshold = options.r2Threshold ?? 0.9
  const layers = kan.layers.map((layer, l) => {
    const mask = toFlat(((state.buffers[String(l)] ?? {}) as Record<string, unknown>)['mask'] as Value)
    const bias = layer.hasBias
      ? Array.from(toFlat(((state.params[String(l)] ?? {}) as Record<string, unknown>)['bias'] as Value))
      : new Array<number>(layer.outFeatures).fill(0)
    const edges = Array.from({ length: layer.outFeatures }, (_, o) =>
      Array.from({ length: layer.inFeatures }, (__, i): SymbolicEdge | undefined => {
        if ((mask[o * layer.inFeatures + i] as number) === 0) return null
        const best = suggestSymbolic(kan, state, x, l, i, o, options)[0]
        if (best === undefined || best.r2 < threshold) return undefined
        const { score: _s, ...fit } = best
        return fit
      }),
    )
    return { edges, bias }
  })
  return { widths: [...kan.widths], layers }
}

const edgeValue = (e: SymbolicEdge, v: number): number => {
  if (e === null) return 0
  const f = symbolicFunctions[e.name] as SymbolicFunction
  return e.a * f.f(e.b * v + e.c) + e.d
}

/**
 * Evaluate a symbolic network on rows: the numbers the formula of {@link symbolicFormula} gives.
 *
 * @param net - The symbolic network (every edge fixed).
 * @param x - Inputs `[n, n_in]`.
 * @returns The outputs, one row per input row.
 * @throws OmaiError `value` when an edge is not fixed.
 * @concept symbolic-regression
 * @example
 * ```ts
 * import { evaluateSymbolic } from 'omai-engine/nn'
 * const net = { widths: [1, 1], layers: [{ edges: [[{ name: 'x^2', a: 2, b: 1, c: 0, d: 0, r2: 1 }]], bias: [0] }] }
 * evaluateSymbolic(net, [[3]]) // => [[18]]
 * ```
 */
export function evaluateSymbolic(net: SymbolicKan, x: TensorLike): number[][] {
  const t = asTensor(x)
  const [n, d] = [t.shape[0] ?? 0, t.shape[1] ?? 1]
  const flat = toFlat(t)
  return Array.from({ length: n }, (_, r) => {
    let h = Array.from({ length: d }, (__, i) => flat[r * d + i] as number)
    net.layers.forEach((layer, l) => {
      h = layer.edges.map((row, o) =>
        row.reduce<number>((s, e, i) => {
          if (e === undefined)
            throw new OmaiError('value', `evaluateSymbolic: edge ${i} -> ${o} of layer ${l} is not fixed`)
          return s + edgeValue(e, h[i] as number)
        }, layer.bias[o] as number),
      )
    })
    return h
  })
}

/** The closed form of a symbolic network. */
export interface SymbolicFormula {
  /** One expression per output node, in the input names. */
  readonly expressions: readonly Expr[]
  /** The same as text. */
  readonly text: readonly string[]
}

const roundTo = (v: number, digits: number): number => (v === 0 ? 0 : Number(v.toPrecision(digits)))

/**
 * The closed form of a network whose edges are all fixed: every node is the sum of its edges' functions of the nodes
 * below plus its bias, composed down to the inputs (named `x_1 ... x_n` unless given). The constants are rounded to
 * `digits` significant digits (full precision when `digits` is 0).
 *
 * @param net - The symbolic network (see {@link fixSymbolic}).
 * @param names - The names of the inputs.
 * @param digits - Significant digits of the constants (default 4).
 * @returns The expressions and their text.
 * @throws OmaiError `value` when an edge is not fixed.
 * @concept symbolic-regression
 * @example
 * ```ts
 * import { symbolicFormula } from 'omai-engine/nn'
 * const net = { widths: [1, 1], layers: [{ edges: [[{ name: 'x^2', a: 2, b: 1, c: 0, d: 0, r2: 1 }]], bias: [0] }] }
 * symbolicFormula(net, ['t']).text // => ['2*t**2']
 * ```
 */
export function symbolicFormula(net: SymbolicKan, names?: readonly string[], digits = 4): SymbolicFormula {
  const r = (v: number): Expr => num(digits > 0 ? roundTo(v, digits) : v)
  let h: Expr[] = Array.from({ length: net.widths[0] as number }, (_, i) => sym(names?.[i] ?? `x_${i + 1}`))
  net.layers.forEach((layer, l) => {
    h = layer.edges.map((row, o) => {
      const terms: Expr[] = []
      if ((layer.bias[o] as number) !== 0) terms.push(r(layer.bias[o] as number))
      row.forEach((e, i) => {
        if (e === undefined)
          throw new OmaiError('value', `symbolicFormula: edge ${i} -> ${o} of layer ${l} is not fixed`)
        if (e === null || e.name === '0') return
        const f = symbolicFunctions[e.name] as SymbolicFunction
        const inner = addExpr(mulExpr(r(e.b), h[i] as Expr), r(e.c))
        const body = e.name === 'x' ? inner : f.expr(inner)
        terms.push(addExpr(mulExpr(r(e.a), body), r(e.d)))
      })
      return terms.length === 0 ? num(0) : addExpr(...terms)
    })
  })
  return { expressions: h, text: h.map((e) => toText(e)) }
}
