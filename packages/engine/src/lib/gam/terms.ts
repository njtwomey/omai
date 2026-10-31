/**
 * The terms of a generalised additive model and the penalised design they make: smooths of one or several covariates
 * on any basis of `stats/smoothing` (`s`), tensor-product smooths and interactions (`te`, `ti`), parametric slopes
 * (`linear`) and factors, unpenalised dummy columns or random effects (`factor`). Each term becomes a block of design
 * columns with its penalty matrices and null-space dimension; smooths are centred (sum-to-zero) so the intercept is
 * identifiable, and a term can be evaluated at new data from what it learned on the training data.
 */

import { MdmlError } from '../base/index.ts'
import { type MatrixView, matrixView } from '../learn/index.ts'
import { dgemm, dsyev } from '../numerics/linalg/index.ts'
import {
  bs,
  cc,
  cr,
  duchon,
  gp,
  ti as interactionBasis,
  ns,
  ps,
  type SmoothBasis,
  sumToZeroConstraint,
  te as tensorBasis,
  tp,
  ts,
} from '../stats/smoothing/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type ConstraintBlock, type ShapeKind, shapeRows } from './shape.ts'

/** The bases a smooth term can use (`stats/smoothing`): the names are mgcv's. */
export type SmoothKind = 'cr' | 'cc' | 'ps' | 'bs' | 'ns' | 'tp' | 'ts' | 'duchon' | 'gp'

/** Options of a smooth term {@link s}. */
export interface SmoothOptions {
  /** The basis (default `cr` for one covariate, `tp` for several). */
  readonly basis?: SmoothKind
  /** The basis dimension before centring (default 10 for one covariate, 30 for several, at most the distinct values). */
  readonly k?: number
  /** The degree of `ps` and `bs` bases (default 3). */
  readonly degree?: number
  /** The difference penalty order of `ps` (default 2). */
  readonly order?: number
  /** A covariate (column index) the smooth is multiplied by: a varying-coefficient term `f(x) z`. */
  readonly by?: number
  /** The label (default `s(x0)`). */
  readonly label?: string
  /**
   * Constrain the effect's shape: `increasing`, `decreasing`, `convex`, `concave`, or several (`['increasing',
   * 'concave']`). One covariate only. The fit is then a quadratic program per step ({@link ShapeKind}).
   */
  readonly shape?: ShapeKind | readonly ShapeKind[]
  /** A lower bound on the (centred) partial effect, imposed on a grid over the covariate's range. */
  readonly lower?: number
  /** An upper bound on the (centred) partial effect. */
  readonly upper?: number
  /**
   * Add a penalty on the null space of the smoothing penalty (Marra and Wood's double penalty, mgcv's `select = TRUE`), so
   * the term can be shrunk out of the model by the smoothing-parameter criterion.
   */
  readonly select?: boolean
}

/** Options of a tensor-product term {@link te} or {@link ti}. */
export interface TensorOptions {
  /** The marginal basis, one for all or one per margin (default `cr`). */
  readonly basis?: SmoothKind | readonly SmoothKind[]
  /** The marginal dimension, one for all or one per margin (default 5). */
  readonly k?: number | readonly number[]
  readonly label?: string
}

/** A term of the additive predictor. */
export type GamTerm =
  | {
      readonly kind: 'smooth'
      readonly features: readonly number[]
      readonly basis: SmoothKind
      readonly k: number | undefined
      readonly degree: number
      readonly order: number
      readonly by: number | undefined
      readonly label: string
      readonly shape: readonly ShapeKind[]
      readonly lower: number | undefined
      readonly upper: number | undefined
      readonly select: boolean
    }
  | {
      readonly kind: 'te' | 'ti'
      readonly features: readonly number[]
      readonly bases: readonly SmoothKind[]
      readonly ks: readonly number[]
      readonly label: string
    }
  | { readonly kind: 'linear'; readonly features: readonly [number]; readonly label: string }
  | {
      readonly kind: 'factor'
      readonly features: readonly [number]
      readonly randomEffect: boolean
      readonly label: string
    }

const names = (features: readonly number[]): string => features.map((f) => `x${f}`).join(',')

/**
 * A smooth term of one or several covariates (mgcv's `s`): a penalised regression spline whose wiggliness is chosen
 * by the smoothing-parameter criterion.
 *
 * @param feature - The column of the covariate, or the columns of an isotropic smooth of several.
 * @param options - The basis, its dimension and degree, a `by` covariate and the label.
 * @returns The term.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { s } from 'mdml-engine/gam'
 * s(0, { basis: 'ps', k: 8 }).label // => 's(x0)'
 * s([0, 1]).basis // => 'tp'
 * ```
 */
export function s(feature: number | readonly number[], options: SmoothOptions = {}): GamTerm {
  const features = typeof feature === 'number' ? [feature] : [...feature]
  if (features.length === 0) throw new MdmlError('value', 's: needs at least one covariate')
  return {
    kind: 'smooth',
    features,
    basis: options.basis ?? (features.length === 1 ? 'cr' : 'tp'),
    k: options.k,
    degree: options.degree ?? 3,
    order: options.order ?? 2,
    by: options.by,
    label: options.label ?? `s(${names(features)})${options.by === undefined ? '' : `:x${options.by}`}`,
    shape: options.shape === undefined ? [] : typeof options.shape === 'string' ? [options.shape] : [...options.shape],
    lower: options.lower,
    upper: options.upper,
    select: options.select ?? false,
  }
}

function tensorTerm(kind: 'te' | 'ti', features: readonly number[], options: TensorOptions): GamTerm {
  if (features.length < 2) throw new MdmlError('value', `${kind}: needs at least two covariates`)
  const per = <T>(v: T | readonly T[] | undefined, fallback: T): T[] =>
    Array.isArray(v) ? [...(v as readonly T[])] : features.map(() => (v === undefined ? fallback : (v as T)))
  const bases = per<SmoothKind>(options.basis, 'cr')
  const ks = per<number>(options.k, 5)
  if (bases.length !== features.length || ks.length !== features.length) {
    throw new MdmlError('shape', `${kind}: basis and k need one entry per covariate`)
  }
  return { kind, features: [...features], bases, ks, label: options.label ?? `${kind}(${names(features)})` }
}

/**
 * A tensor-product smooth (mgcv's `te`): the marginal bases multiplied out, with one smoothing parameter per margin,
 * so the covariates may be on different scales.
 *
 * @param features - The columns of the covariates.
 * @param options - The marginal bases and dimensions.
 * @returns The term.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { te } from 'mdml-engine/gam'
 * te([0, 1], { k: [5, 4] }).label // => 'te(x0,x1)'
 * ```
 */
export function te(features: readonly number[], options: TensorOptions = {}): GamTerm {
  return tensorTerm('te', features, options)
}

/**
 * A tensor-product interaction (mgcv's `ti`): a tensor product whose margins' main effects are removed, to be added
 * to the smooths of the covariates themselves.
 *
 * @param features - The columns of the covariates.
 * @param options - The marginal bases and dimensions.
 * @returns The term.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { ti } from 'mdml-engine/gam'
 * ti([0, 1]).kind // => 'ti'
 * ```
 */
export function ti(features: readonly number[], options: TensorOptions = {}): GamTerm {
  return tensorTerm('ti', features, options)
}

/**
 * A parametric slope: the covariate enters linearly and unpenalised.
 *
 * @param feature - The column.
 * @param label - The label (default `x0`).
 * @returns The term.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { linear } from 'mdml-engine/gam'
 * linear(2).label // => 'x2'
 * ```
 */
export function linear(feature: number, label?: string): GamTerm {
  return { kind: 'linear', features: [feature], label: label ?? `x${feature}` }
}

/**
 * A factor: the column's distinct values are the levels. Unpenalised, it is treatment-coded (the first level is the
 * reference); as a random effect (`randomEffect: true`) it has one penalised column per level, the smoothing parameter
 * being the ratio of the residual to the random-effect variance.
 *
 * @param feature - The column holding the level.
 * @param options - `randomEffect` and the label.
 * @returns The term.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { factor } from 'mdml-engine/gam'
 * factor(1, { randomEffect: true }).label // => 'factor(x1)'
 * ```
 */
export function factor(
  feature: number,
  options: { readonly randomEffect?: boolean; readonly label?: string } = {},
): GamTerm {
  return {
    kind: 'factor',
    features: [feature],
    randomEffect: options.randomEffect ?? false,
    label: options.label ?? `factor(x${feature})`,
  }
}

/** One penalty of a term: a `width x width` matrix, row-major. */
export interface TermPenalty {
  readonly matrix: Float64Array
}

/** A term compiled against training data: its columns, penalties, and how to evaluate it elsewhere. */
export interface CompiledTerm {
  readonly term: GamTerm
  readonly label: string
  /** The first column of the block in the design (column 0 is the intercept). */
  readonly offset: number
  readonly width: number
  readonly penalties: readonly TermPenalty[]
  /** The dimension of the part of the block no penalty touches (the term's unpenalised, parametric part). */
  readonly nullSpaceDimension: number
  /** Shape and bound constraints on the term's coefficients `rows beta >= rhs`, when it has any. */
  readonly constraints?: ConstraintBlock
  /** The block `[m, width]`, row-major, at the rows of a matrix. */
  evaluate(x: MatrixView): Float64Array
  /**
   * The exact first or second derivative of the block in the term's covariate `[m, width]`, for a smooth of one covariate
   * on a basis with a closed-form derivative (`ps`, `bs`, `cr`, `cc`, `tp`, `ts`, `duchon`, `gp`), centred and scaled by
   * its `by` covariate like `evaluate`.
   */
  derivative?(x: MatrixView, order: 1 | 2): Float64Array
}

/** A penalised design: the intercept and the blocks of the terms. */
export interface GamDesign {
  readonly n: number
  /** The columns `p`, the intercept included. */
  readonly p: number
  readonly nFeatures: number
  /** The design matrix `[n, p]`, row-major. */
  readonly X: Float64Array
  readonly terms: readonly CompiledTerm[]
  /** The number of smoothing parameters: one per penalty, in the order of the terms. */
  readonly nPenalties: number
  /** The constraints of all terms `rows beta >= rhs` over the design's columns, when any term has some. */
  readonly constraints?: ConstraintBlock
}

function columnsOf(m: MatrixView, features: readonly number[]): Tensor {
  for (const f of features) {
    if (!Number.isInteger(f) || f < 0 || f >= m.d) {
      throw new MdmlError('shape', `gam: term refers to column ${f} but x has ${m.d}`)
    }
  }
  const out = new Float64Array(m.n * features.length)
  for (let i = 0; i < m.n; i++) {
    features.forEach((f, c) => {
      out[i * features.length + c] = m.data[i * m.d + f] ?? 0
    })
  }
  return fromStorage(out, features.length === 1 ? [m.n] : [m.n, features.length], 'float64')
}

function distinct(values: ArrayLike<number>): number {
  return new Set(Array.from(values)).size
}

function marginal(kind: SmoothKind, x: TensorLike, k: number, degree: number, order: number): SmoothBasis {
  switch (kind) {
    case 'cr':
      return cr(x, { k })
    case 'cc':
      return cc(x, { k })
    case 'ps':
      return ps(x, { k, degree, order })
    case 'bs':
      return bs(x, { df: k, degree })
    case 'ns':
      return ns(x, { df: k })
    case 'tp':
      return tp(x, { k })
    case 'ts':
      return ts(x, { k })
    case 'duchon':
      return duchon(x, { k })
    case 'gp':
      return gp(x, { k })
  }
}

/** Centre a basis by absorbing its sum-to-zero constraint: columns `X Z`, penalties `Z' S Z`. */
function centre(basis: SmoothBasis): {
  readonly X: Float64Array
  readonly penalties: Float64Array[]
  readonly Z: Float64Array
  readonly k: number
  readonly kc: number
  readonly nullDim: number
} {
  const n = basis.X.shape[0] ?? 0
  const k = basis.X.shape[1] ?? 0
  const Z = Float64Array.from(toFlat(sumToZeroConstraint(basis.X)))
  const kc = k - 1
  const X = dgemm(toFlat(basis.X), Z, n, k, kc)
  const penalties = basis.penalties.map((S) => {
    const sz = dgemm(toFlat(S), Z, k, k, kc)
    return dgemm(Z, sz, kc, k, kc, { transposeA: true })
  })
  return { X, penalties, Z, k, kc, nullDim: Math.max(0, basis.nullSpaceDimension - 1) }
}

/** The projector on the null space of the sum of penalty matrices (the double penalty), or undefined if it is empty. */
function nullSpacePenalty(penalties: readonly Float64Array[], width: number): Float64Array | undefined {
  const total = new Float64Array(width * width)
  for (const pen of penalties) for (let i = 0; i < total.length; i++) total[i] = (total[i] ?? 0) + (pen[i] ?? 0)
  const { w, v } = dsyev(total, width)
  const top = Math.max(...Array.from(w), 1e-300)
  const out = new Float64Array(width * width)
  let any = false
  w.forEach((value, c) => {
    if (value > 1e-9 * top) return
    any = true
    for (let i = 0; i < width; i++)
      for (let j = 0; j < width; j++)
        out[i * width + j] = (out[i * width + j] ?? 0) + (v[i * width + c] ?? 0) * (v[j * width + c] ?? 0)
  })
  return any ? out : undefined
}

function compile(term: GamTerm, m: MatrixView, offset: number): CompiledTerm {
  const flat = (t: Tensor): Float64Array => Float64Array.from(toFlat(t))
  switch (term.kind) {
    case 'smooth': {
      const x = columnsOf(m, term.features)
      const d = term.features.length
      const distinctValues = d === 1 ? distinct(toFlat(x)) : m.n
      const k = term.k ?? Math.max(3, Math.min(d === 1 ? 10 : 30, distinctValues))
      const basis = marginal(term.basis, x, k, term.degree, term.order)
      const byColumn = term.by === undefined ? undefined : Array.from(columnsOf(m, [term.by]).data)
      // A varying-coefficient term f(x) z is not centred: the intercept does not confound it.
      const c = byColumn === undefined ? centre(basis) : undefined
      const kRaw = basis.X.shape[1] ?? 0
      const width = c === undefined ? kRaw : c.kc
      const penalties = (c?.penalties ?? basis.penalties.map(flat)).map((matrix) => ({ matrix }))
      if (term.select) {
        const extra = nullSpacePenalty(
          penalties.map((p) => p.matrix),
          width,
        )
        if (extra !== undefined) penalties.push({ matrix: extra })
      }
      const scaleBy = (block: Float64Array, rows: number, by: ArrayLike<number> | undefined): Float64Array => {
        if (by === undefined) return block
        for (let i = 0; i < rows; i++)
          for (let j = 0; j < width; j++) block[i * width + j] = (block[i * width + j] ?? 0) * (by[i] ?? 0)
        return block
      }
      const evaluate = (q: MatrixView): Float64Array => {
        const block = flat(basis.evaluate(columnsOf(q, term.features)))
        const centred = c === undefined ? block : dgemm(block, c.Z, q.n, c.k, c.kc)
        return scaleBy(centred, q.n, term.by === undefined ? undefined : columnsOf(q, [term.by]).data)
      }
      const basisDerivative = basis.derivative
      const derivative =
        basisDerivative === undefined || d !== 1
          ? undefined
          : (q: MatrixView, order: 1 | 2): Float64Array => {
              const block = flat(basisDerivative(columnsOf(q, term.features), order))
              const centred = c === undefined ? block : dgemm(block, c.Z, q.n, c.k, c.kc)
              return scaleBy(centred, q.n, term.by === undefined ? undefined : columnsOf(q, [term.by]).data)
            }
      const constrained = term.shape.length > 0 || term.lower !== undefined || term.upper !== undefined
      let constraints: ConstraintBlock | undefined
      if (constrained) {
        if (d !== 1 || term.by !== undefined) {
          throw new MdmlError(
            'value',
            `gam: ${term.label}: shape constraints and bounds need one covariate and no \`by\``,
          )
        }
        const values = Array.from(toFlat(x))
        const lo = Math.min(...values)
        const hi = Math.max(...values)
        const gridSize = Math.min(80, Math.max(30, 3 * kRaw))
        const gridData = new Float64Array(gridSize * m.d)
        for (let i = 0; i < gridSize; i++)
          gridData[i * m.d + (term.features[0] ?? 0)] = lo + ((hi - lo) * i) / (gridSize - 1)
        constraints = shapeRows({
          shapes: term.shape,
          lower: term.lower,
          upper: term.upper,
          basis: term.basis,
          k: kRaw,
          Z: c?.Z,
          width,
          grid: evaluate({ n: gridSize, d: m.d, data: gridData }),
          gridSize,
        })
      }
      return {
        term,
        label: term.label,
        offset,
        width,
        penalties,
        nullSpaceDimension: c === undefined ? basis.nullSpaceDimension : c.nullDim,
        ...(constraints === undefined ? {} : { constraints }),
        evaluate,
        ...(derivative === undefined ? {} : { derivative }),
      }
    }
    case 'te':
    case 'ti': {
      const margins = term.features.map((f, i) => {
        const col = columnsOf(m, [f])
        const cap = Math.max(3, Math.min(term.ks[i] ?? 5, distinct(toFlat(col))))
        return marginal(term.bases[i] ?? 'cr', col, cap, 3, 2)
      })
      const basis = term.kind === 'te' ? tensorBasis(margins) : interactionBasis(margins)
      const k = basis.X.shape[1] ?? 0
      const c = term.kind === 'te' ? centre(basis) : undefined
      const width = c === undefined ? k : c.kc
      return {
        term,
        label: term.label,
        offset,
        width,
        penalties: (c?.penalties ?? basis.penalties.map(flat)).map((matrix) => ({ matrix })),
        nullSpaceDimension: c === undefined ? basis.nullSpaceDimension : c.nullDim,
        evaluate: (q) => {
          const block = flat(basis.evaluate(columnsOf(q, term.features)))
          return c === undefined ? block : dgemm(block, c.Z, q.n, c.k, c.kc)
        },
      }
    }
    case 'linear': {
      const f = term.features[0]
      return {
        term,
        label: term.label,
        offset,
        width: 1,
        penalties: [],
        nullSpaceDimension: 1,
        evaluate: (q) => Float64Array.from(columnsOf(q, [f]).data),
      }
    }
    case 'factor': {
      const f = term.features[0]
      const levels = Array.from(new Set(Array.from(columnsOf(m, [f]).data))).sort((a, b) => a - b)
      const indicator = (q: MatrixView): Float64Array => {
        const values = columnsOf(q, [f]).data
        const out = new Float64Array(q.n * levels.length)
        for (let i = 0; i < q.n; i++) {
          const at = levels.indexOf(values[i] ?? Number.NaN)
          if (at < 0) throw new MdmlError('value', `gam: ${term.label} has a level (${values[i]}) not seen in training`)
          out[i * levels.length + at] = 1
        }
        return out
      }
      if (term.randomEffect) {
        const w = levels.length
        return {
          term,
          label: term.label,
          offset,
          width: w,
          penalties: [{ matrix: Float64Array.from({ length: w * w }, (_, i) => (i % (w + 1) === 0 ? 1 : 0)) }],
          nullSpaceDimension: 0,
          evaluate: indicator,
        }
      }
      const w = levels.length - 1
      return {
        term,
        label: term.label,
        offset,
        width: w,
        penalties: [],
        nullSpaceDimension: w,
        evaluate: (q) => {
          const full = indicator(q)
          const out = new Float64Array(q.n * w)
          for (let i = 0; i < q.n; i++)
            for (let j = 0; j < w; j++) out[i * w + j] = full[i * levels.length + j + 1] ?? 0
          return out
        },
      }
    }
  }
}

/**
 * Build the penalised design of a list of terms on training data: the intercept, then each term's block, with the
 * penalties and what is needed to evaluate the blocks at new rows ({@link evaluateDesign}).
 *
 * @param x - The covariates `[n, d]`.
 * @param terms - The terms (default one `s` per column).
 * @returns The design.
 * @throws MdmlError `shape` for a term that refers to a missing column.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { buildDesign, linear, s } from 'mdml-engine/gam'
 * const x = [[0, 1], [1, 3], [2, 2], [3, 5], [4, 4], [5, 6]]
 * const design = buildDesign(x, [s(0, { k: 4 }), linear(1)])
 * [design.p, design.nPenalties] // => [5, 1]
 * ```
 */
export function buildDesign(x: TensorLike, terms?: readonly GamTerm[]): GamDesign {
  const m = matrixView(x, 'gam')
  const list = terms ?? Array.from({ length: m.d }, (_, j) => s(j))
  const compiled: CompiledTerm[] = []
  let offset = 1
  for (const term of list) {
    const c = compile(term, m, offset)
    compiled.push(c)
    offset += c.width
  }
  const withConstraints = compiled.filter((t) => t.constraints !== undefined)
  let constraints: ConstraintBlock | undefined
  if (withConstraints.length > 0) {
    const count = withConstraints.reduce((a, t) => a + (t.constraints?.count ?? 0), 0)
    const rows = new Float64Array(count * offset)
    const rhs = new Float64Array(count)
    let at = 0
    for (const t of withConstraints) {
      const block = t.constraints as ConstraintBlock
      for (let r = 0; r < block.count; r++) {
        for (let j = 0; j < block.width; j++)
          rows[(at + r) * offset + t.offset + j] = block.rows[r * block.width + j] ?? 0
        rhs[at + r] = block.rhs[r] ?? 0
      }
      at += block.count
    }
    constraints = { rows, rhs, count, width: offset }
  }
  const design: GamDesign = {
    n: m.n,
    p: offset,
    nFeatures: m.d,
    X: new Float64Array(m.n * offset),
    terms: compiled,
    nPenalties: compiled.reduce((a, t) => a + t.penalties.length, 0),
    ...(constraints === undefined ? {} : { constraints }),
  }
  fillDesign(design, m, design.X)
  return design
}

function fillDesign(design: GamDesign, m: MatrixView, out: Float64Array): void {
  const p = design.p
  for (let i = 0; i < m.n; i++) out[i * p] = 1
  for (const t of design.terms) {
    const block = t.evaluate(m)
    for (let i = 0; i < m.n; i++)
      for (let j = 0; j < t.width; j++) out[i * p + t.offset + j] = block[i * t.width + j] ?? 0
  }
}

/**
 * The design matrix of a built design at new rows.
 *
 * @param design - A design built on training data.
 * @param x - New covariates `[m, d]`.
 * @returns The matrix `[m, p]` and its row count.
 * @throws MdmlError `shape` for a wrong column count; `value` for a factor level not seen in training.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { buildDesign, evaluateDesign, linear } from 'mdml-engine/gam'
 * const design = buildDesign([[1], [2], [3]], [linear(0)])
 * Array.from(evaluateDesign(design, [[10]]).X) // => [1, 10]
 * ```
 */
export function evaluateDesign(design: GamDesign, x: TensorLike): { readonly X: Float64Array; readonly n: number } {
  const m = matrixView(x, 'gam', { nFeatures: design.nFeatures })
  const out = new Float64Array(m.n * design.p)
  fillDesign(design, m, out)
  return { X: out, n: m.n }
}
