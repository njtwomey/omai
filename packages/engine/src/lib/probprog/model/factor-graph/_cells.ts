/**
 * The symbolic values the model compiler computes with: tensors (`Sym`) of cells, each cell an affine form over the graph's
 * random scalars (`Aff` over `Atom`s) or one of the few other kinds the idioms need (a variable of another family, the
 * stick-breaking weights, a scaled or reciprocal gamma variable, a known function of a discrete variable as a `lookup`).
 * The arithmetic of each kind is here as pure functions that never touch a graph: `combine`, `scaleAff`, `affOf`,
 * `lookupArithmetic`, `recipArithmetic`, `scaledArithmetic`. A rule that does not hold throws a `Refusal`.
 */

import type { FamilyKind } from '../../../message-passing/index.ts'
import {
  inverseGammaArithmetic,
  lookupNotGaussian,
  nonGaussianArithmetic,
  opOfDiscrete,
  opOfDiscreteAndRandom,
  opOfGammaAndRandom,
  opOfInverseGamma,
  opOfInverseGammaAndRandom,
  opOfPartlyRandomLookup,
  opOfTwoDiscretes,
  scaledPrecisionArithmetic,
  stickBreakingArithmetic,
} from './_reasons.ts'

/** One random scalar of the graph an affine form is made of: a variable, or a coordinate of a vector variable. */
export type Atom =
  | { readonly kind: 'var'; readonly name: string }
  | { readonly kind: 'comp'; readonly vector: string; readonly index: number; readonly size: number }

/** An affine form `offset + sum of coefficient times atom`; a constant has no terms. */
export interface Aff {
  readonly terms: ReadonlyMap<string, number>
  readonly offset: number
}

/** An element of a symbolic tensor: an affine form of Gaussian atoms, or a variable of another family (not arithmetic). */
export type Cell =
  | { readonly kind: 'aff'; readonly aff: Aff }
  | { readonly kind: 'rv'; readonly variable: string; readonly family: FamilyKind }
  // The stick-breaking weights of beta variables, `v_k prod_{j<k} (1 - v_j)` and the remainder (one more weight than breaks): the
  // probabilities of a categorical, which reads them through the `stick-breaking` factor (the truncated Dirichlet process).
  | { readonly kind: 'sticks'; readonly variables: readonly string[] }
  // A known multiple of a gamma variable plus a known offset, `scale tau + offset`: the precision of a Gaussian that shares
  // one smoothing parameter or one global scale (it has no other use: it is not a Gaussian value).
  | { readonly kind: 'scaled'; readonly variable: string; readonly scale: number; readonly offset: number }
  // A known multiple of the reciprocal of a gamma variable, `scale / tau`: an inverse-gamma variance (the variable is its
  // precision), which is mapped only as the square of the scale of a Gaussian (the precision idiom).
  | { readonly kind: 'recip'; readonly variable: string; readonly scale: number }
  // A known function of a discrete variable, one number per state (`at(means, z)`, `z * 6 - 3`): a value that is not Gaussian
  // but selects among known ones, which a likelihood or a prior reads through a table or a gate.
  // `items[k]`, when present, is a random Gaussian form that state k takes in place of the known `values[k]` (a mixture of a known
  // value and a random one: the clutter problem's `where(z, 0, theta)`).
  // `precisions[k]`, when present, is the gamma precision `scale tau + offset` that state k takes (a precision per component of a
  // mixture, read by a Gaussian observation as the gate of `gaussian-gamma-precision` branches); `values[k]` is then the known
  // precision of a state that has none.
  | {
      readonly kind: 'lookup'
      readonly variable: string
      readonly values: readonly number[]
      readonly items?: readonly (Aff | undefined)[]
      readonly precisions?: readonly (Precision | undefined)[]
    }

/** A gamma variable's multiple plus an offset: the precision `scale tau + offset`. */
export interface Precision {
  readonly variable: string
  readonly scale: number
  readonly offset: number
}

/** A tensor over named dimensions whose elements are cells, row-major in the order of `dims`. */
export interface Sym {
  readonly dims: readonly string[]
  readonly cells: readonly Cell[]
}

export const constAff = (offset: number): Aff => ({ terms: new Map(), offset })
export const constCell = (v: number): Cell => ({ kind: 'aff', aff: constAff(v) })
export const isConstCell = (c: Cell): c is Cell & { kind: 'aff' } => c.kind === 'aff' && c.aff.terms.size === 0

export function combine(a: Aff, b: Aff, kb: number): Aff {
  const terms = new Map(a.terms)
  for (const [k, c] of b.terms) {
    const v = (terms.get(k) ?? 0) + kb * c
    if (v === 0) terms.delete(k)
    else terms.set(k, v)
  }
  return { terms, offset: a.offset + kb * b.offset }
}

export function scaleAff(a: Aff, k: number): Aff {
  if (k === 0) return constAff(0)
  return { terms: new Map([...a.terms].map(([key, c]) => [key, c * k])), offset: a.offset * k }
}

export function affOf(c: Cell): Aff {
  if (c.kind === 'lookup') throw lookupNotGaussian(c.variable)
  if (c.kind === 'scaled') throw scaledPrecisionArithmetic(c.variable)
  if (c.kind === 'recip') throw inverseGammaArithmetic(c.variable)
  if (c.kind === 'sticks') throw stickBreakingArithmetic()
  if (c.kind !== 'aff') throw nonGaussianArithmetic(c.family, c.variable)
  return c.aff
}

/** A gamma variable or a known multiple of one plus an offset, as `scale tau + offset`; undefined for any other cell. */
export function scaledOf(
  c: Cell,
): { readonly variable: string; readonly scale: number; readonly offset: number } | undefined {
  if (c.kind === 'scaled') return c
  if (c.kind === 'rv' && c.family === 'gamma') return { variable: c.variable, scale: 1, offset: 0 }
  return undefined
}

/**
 * The arithmetic of a known function of a discrete variable: a discrete or Bernoulli variable (its states as numbers) or a
 * lookup, combined with a known value or with another lookup of the same variable, state by state. Undefined when neither
 * operand is one, so the other rules apply.
 */
export function lookupArithmetic(op: string, x: Cell, y: Cell, states: (variable: string) => number): Cell | undefined {
  const asLookup = (c: Cell): (Cell & { kind: 'lookup' }) | undefined => {
    if (c.kind === 'lookup') return c
    if (c.kind === 'rv' && (c.family === 'discrete' || c.family === 'bernoulli')) {
      const k = c.family === 'bernoulli' ? 2 : states(c.variable)
      return lookup(
        c.variable,
        Array.from({ length: k }, (_, i) => i),
      )
    }
    return undefined
  }
  const a = asLookup(x)
  const b = asLookup(y)
  if (a === undefined && b === undefined) return undefined
  if ((a?.items ?? b?.items)?.some((v) => v !== undefined)) throw opOfPartlyRandomLookup(op)
  const f = (u: number, v: number): number =>
    op === 'add' ? u + v : op === 'sub' ? u - v : op === 'mul' ? u * v : u / v
  if (!['add', 'sub', 'mul', 'div'].includes(op)) throw opOfDiscrete(op)
  if (a !== undefined && b !== undefined) {
    if (a.variable !== b.variable) throw opOfTwoDiscretes(op, a.variable, b.variable)
    return lookup(
      a.variable,
      a.values.map((u, i) => f(u, b.values[i] as number)),
    )
  }
  const [look, other, flip] = a !== undefined ? [a, y, false] : [b as Cell & { kind: 'lookup' }, x, true]
  if (!isConstCell(other)) throw opOfDiscreteAndRandom(op)
  const k = other.aff.offset
  return lookup(
    look.variable,
    look.values.map((u) => (flip ? f(k, u) : f(u, k))),
  )
}

/**
 * The arithmetic of an inverse-gamma variance `scale / tau`: a known multiple (`mul`, `div` by a known value) stays one; a known
 * value over it is the precision `value tau / scale`, a scaled gamma variable. Undefined when neither operand is one.
 */
export function recipArithmetic(op: string, x: Cell, y: Cell): Cell | undefined {
  if (x.kind !== 'recip' && y.kind !== 'recip') return undefined
  const other = x.kind === 'recip' ? y : x
  const r = (x.kind === 'recip' ? x : y) as Cell & { kind: 'recip' }
  const known = isConstCell(other) ? other.aff.offset : undefined
  if (known === undefined || (x.kind === 'recip' && y.kind === 'recip') || !(known > 0))
    throw opOfInverseGammaAndRandom(op)
  if (op === 'mul') return { kind: 'recip', variable: r.variable, scale: r.scale * known }
  if (op === 'div' && x.kind === 'recip') return { kind: 'recip', variable: r.variable, scale: r.scale / known }
  if (op === 'div') return { kind: 'scaled', variable: r.variable, scale: known / r.scale, offset: 0 }
  throw opOfInverseGamma(op)
}

/**
 * The arithmetic of a precision: a gamma variable (or a scaled one) times, plus, minus or over a known value. Undefined when
 * neither operand is a gamma-based cell, so the ordinary affine rules apply.
 */
export function scaledArithmetic(op: string, x: Cell, y: Cell): Cell | undefined {
  const left = scaledOf(x)
  const right = scaledOf(y)
  if (left === undefined && right === undefined) return undefined
  const other = left === undefined ? x : y
  const known = isConstCell(other) ? other.aff.offset : undefined
  const g = (left ?? right) as { variable: string; scale: number; offset: number }
  const refuse = (): never => {
    throw opOfGammaAndRandom(op)
  }
  if (known === undefined || (left !== undefined && right !== undefined)) return refuse()
  switch (op) {
    case 'mul':
      return { kind: 'scaled', variable: g.variable, scale: g.scale * known, offset: g.offset * known }
    case 'div':
      if (left === undefined) return refuse()
      return { kind: 'scaled', variable: g.variable, scale: g.scale / known, offset: g.offset / known }
    case 'add':
      return { kind: 'scaled', variable: g.variable, scale: g.scale, offset: g.offset + known }
    case 'sub':
      if (left === undefined) return refuse()
      return { kind: 'scaled', variable: g.variable, scale: g.scale, offset: g.offset - known }
    default:
      return refuse()
  }
}

/** A known function of a discrete variable, one number per state; `items` and `precisions` are the random forms some states carry (see `Cell`). */
export function lookup(
  variable: string,
  values: readonly number[],
  extra: {
    readonly items?: readonly (Aff | undefined)[]
    readonly precisions?: readonly (Precision | undefined)[]
  } = {},
): Cell & { readonly kind: 'lookup' } {
  return { kind: 'lookup', variable, values, ...extra }
}

/**
 * The lookup of Gaussian forms, one per state of the discrete variable: known numbers when every form is a constant, else a
 * lookup whose random states carry their forms in `items` (a gate of Gaussian branches reads them).
 */
export function lookupOfForms(variable: string, components: readonly Cell[]): Cell {
  const values = components.map((c) => (isConstCell(c) ? c.aff.offset : 0))
  if (components.every(isConstCell)) return lookup(variable, values)
  return lookup(variable, values, {
    items: components.map((c) => (isConstCell(c) ? undefined : (c as Cell & { kind: 'aff' }).aff)),
  })
}

export const prod = (xs: readonly number[]): number => xs.reduce((a, b) => a * b, 1)

/** The hidden symbol of the log of a positive latent carried on its log (not a node of the spec, so no site can clash with it). */
export const logName = (site: string): string => `${site}#log`

export function unflatten(flat: number, shape: readonly number[]): number[] {
  const idx: number[] = new Array<number>(shape.length).fill(0)
  let rest = flat
  for (let i = shape.length - 1; i >= 0; i--) {
    const n = shape[i] as number
    idx[i] = rest % n
    rest = Math.floor(rest / n)
  }
  return idx
}
