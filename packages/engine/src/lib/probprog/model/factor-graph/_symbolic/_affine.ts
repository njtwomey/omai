/**
 * The evaluator of expressions: an expression as a tensor of affine forms (`evalSym`), by the registry where it is known and
 * symbolically where it is random. A custom function that is affine in one random vector is read as a matrix, `matvec` and `dot`
 * of a known matrix are linear maps of a vector variable, and the operators of the value algebra (`symOp`: arithmetic, sums and
 * means, `matmul`, comparisons, the unary functions, the stick-breaking weights) combine the cells of their operands. Evaluating
 * emits graph structure where a random value needs a variable of its own (a product, an `apply`, a gate: `_functions.ts`,
 * `_gates.ts`), so the order of evaluation is the order of the auxiliary names. `EvalMode` says what a gate keeps.
 */

import type { Json } from '../../../../base/index.ts'
import {
  abs,
  add,
  arange,
  bincount,
  cos,
  eye,
  greater,
  matmul as matmulValue,
  mul,
  stack,
  sub,
  take,
  toArray,
  toFlat,
  type Value,
  zeros,
} from '../../../../tensor/index.ts'
import type { NamedValue } from '../../operators.ts'
import type { ExprSpec } from '../../spec.ts'
import {
  type Aff,
  affOf,
  type Cell,
  combine,
  constAff,
  constCell,
  isConstCell,
  prod,
  type Sym,
  scaleAff,
  scaledOf,
  unflatten,
} from '../_cells.ts'
import type { Context } from '../_context.ts'
import {
  breaksNotBeta,
  customNotAffine,
  customNotSupplied,
  customOpaque,
  customReturnedWrongCount,
  dataNotBound,
  dependsOnDiscrete,
  dependsOnFailed,
  divisionByRandom,
  matmulShapes,
  matrixEntriesVersusVector,
  operatorOnRandom,
  stickBreakingNeedsDimension,
} from '../_reasons.ts'
import { applyAff, applyGamma, clipAff, extremum, mulAff, quotientByChain } from './_functions.ts'
import { compare, gather, pick, special, whereGate } from './_gates.ts'

/**
 * What an evaluation keeps of a value chosen by a discrete variable. `plain`: the `select` gate of the components. `lookup`: the
 * components stay the states of one lookup, which is how an observation whose mean and precision are chosen by the same discrete
 * variable reads them, as one gate over its branches (random means and gamma precisions in each state), not a `select` per value.
 */
export type EvalMode = 'plain' | 'lookup'

/**
 * A custom function of known values and one random vector that is affine in it, `f(z) = f(0) + A z`: the matrix is read off by
 * calling the function on `0` and the unit vectors (and checked at two other points, so a function that is not affine is
 * refused rather than mistaken for one). This is what the whitened Gaussian process `f = L(ell) z` is once the length scale
 * is known: the Cholesky factor is a matrix the graph can use.
 */
export function affineCustom(cx: Context, mode: EvalMode, e: ExprSpec & { readonly kind: 'custom' }): Sym {
  const fn = cx.bound.customs[e.name]
  if (fn === undefined) throw customNotSupplied(e.name)
  const random = e.args.map((a, i) => (cx.isConst(a) ? -1 : i)).filter((i) => i >= 0)
  if (random.length !== 1 || e.eventRank !== 0 || e.dims.length !== 1) throw customOpaque(e.name)
  const at = random[0] as number
  const z = evalSym(cx, e.args[at] as ExprSpec, mode)
  const n = z.cells.length
  const m = cx.size(e.dims[0] as string)
  const known = e.args.map((a, i) => (i === at ? undefined : cx.evalNamed(a)))
  const call = (v: Value): Value => {
    const input = { value: v, dims: z.dims }
    const out = fn(known.map((k, i) => (i === at ? input : (k as NamedValue))))
    const count = toFlat(out).length
    if (count !== m) throw customReturnedWrongCount(e.name, count, m)
    return out
  }
  const f0 = call(zeros([n]))
  // The columns of the matrix: the function at each unit vector, less its value at zero (one row per unit vector).
  const units = eye(n)
  const rows: Value[] = []
  for (let j = 0; j < n; j++) rows.push(sub(call(take(units, j)), f0))
  const jacobian = rows.length === 0 ? zeros([0, m]) : stack(rows)
  for (const probe of [0.37, -1.21]) {
    const point = mul(cos(add(1, mul(2.3, arange(n)))), probe)
    const got = call(point)
    const want = add(f0, matmulValue(point, jacobian))
    const off = sub(got, want)
    const bound = mul(1e-8, add(1, abs(want)))
    if (toFlat(greater(abs(off), bound)).some((flag) => flag !== 0)) throw customNotAffine(e.name)
  }
  cx.handledCustoms.add(e.name)
  const offsets = toFlat(f0)
  const columns = toArray(jacobian) as number[][]
  const cells: Cell[] = []
  for (let i = 0; i < m; i++) {
    let acc = constAff(offsets[i] ?? 0)
    for (let j = 0; j < n; j++) {
      const a = columns[j]?.[i] ?? 0
      if (a !== 0) acc = combine(acc, affOf(z.cells[j] as Cell), a)
    }
    cells.push({ kind: 'aff', aff: acc })
  }
  return { dims: e.dims, cells }
}

/** An expression as a tensor of affine forms, by the registry where it is known and symbolically where it is random. */
export function evalSym(cx: Context, e: ExprSpec, mode: EvalMode = 'plain'): Sym {
  if (cx.isConst(e)) return cx.fromNamed(cx.evalNamed(e))
  switch (e.kind) {
    case 'ref': {
      const s = cx.syms.get(e.name)
      if (s !== undefined) return s
      if (cx.lazy.has(e.name)) throw dependsOnDiscrete(e.name)
      if (cx.unbound.has(e.name)) throw dataNotBound(e.name)
      throw dependsOnFailed(e.name)
    }
    case 'custom':
      return affineCustom(cx, mode, e)
    case 'const':
      return cx.fromNamed(cx.evalNamed(e))
    case 'op':
      if (e.op === 'matvec' || e.op === 'dot') return eventLinear(cx, mode, e)
      return symOp(
        cx,
        mode,
        e.op,
        e.args.map((a) => evalSym(cx, a, mode)),
        e.params ?? {},
      )
  }
}

/**
 * `matvec(A, x)` and `dot(a, x)`: a known matrix or vector applied to a vector `x` (a Gaussian vector variable, its coordinates
 * the cells of the symbol), as affine forms of its coordinates: the step `A x` of a linear dynamical system (a vector of as many
 * forms as `A` has rows), and the scalar reading `a . x`.
 */
export function eventLinear(cx: Context, mode: EvalMode, e: ExprSpec & { readonly kind: 'op' }): Sym {
  const known = cx.evalNamed(e.args[0] as ExprSpec)
  const x = evalSym(cx, e.args[1] as ExprSpec, mode)
  const values = Array.from(toFlat(known.value))
  const n = x.cells.length
  const rows = e.op === 'dot' ? 1 : values.length / n
  if (!Number.isInteger(rows) || rows < 1) throw matrixEntriesVersusVector(e.op, values.length, n)
  const cells: Cell[] = []
  for (let r = 0; r < rows; r++) {
    let acc = constAff(0)
    for (let j = 0; j < n; j++) {
      const a = values[r * n + j] as number
      if (a !== 0) acc = combine(acc, affOf(x.cells[j] as Cell), a)
    }
    cells.push({ kind: 'aff', aff: acc })
  }
  return e.op === 'dot' ? { dims: [], cells } : { dims: x.dims, cells }
}

export function symOp(
  cx: Context,
  mode: EvalMode,
  op: string,
  args: readonly Sym[],
  params: { readonly [k: string]: Json },
): Sym {
  const [a, b] = args as [Sym, Sym]
  switch (op) {
    case 'add':
      return cx.zip(
        [a, b],
        ([x, y]) =>
          special(cx, mode, op, x as Cell, y as Cell) ?? {
            kind: 'aff',
            aff: combine(affOf(x as Cell), affOf(y as Cell), 1),
          },
      )
    case 'sub':
      return cx.zip(
        [a, b],
        ([x, y]) =>
          special(cx, mode, op, x as Cell, y as Cell) ?? {
            kind: 'aff',
            aff: combine(affOf(x as Cell), affOf(y as Cell), -1),
          },
      )
    case 'neg':
      return {
        dims: a.dims,
        cells: a.cells.map((x): Cell => {
          const look = special(cx, mode, 'mul', x, constCell(-1))
          return look?.kind === 'lookup' ? look : { kind: 'aff', aff: scaleAff(affOf(x), -1) }
        }),
      }
    case 'mul':
      return cx.zip(
        [a, b],
        ([x, y], i) =>
          special(cx, mode, op, x as Cell, y as Cell, i) ?? {
            kind: 'aff',
            aff: mulAff(cx, affOf(x as Cell), affOf(y as Cell), i),
          },
      )
    case 'div':
      return cx.zip([a, b], ([x, y], i) => {
        const scaled = special(cx, mode, op, x as Cell, y as Cell)
        if (scaled !== undefined) return scaled
        if (!isConstCell(y as Cell)) {
          // A random divisor that is a function of one latent (`z / sqrt(precision)`, the scale of a non-centred coefficient): the
          // quotient is one `apply-product` factor, the numerator times the reciprocal of the chain behind the divisor.
          const quotient =
            (y as Cell).kind === 'aff' ? quotientByChain(cx, affOf(x as Cell), affOf(y as Cell), i) : undefined
          if (quotient === undefined) throw divisionByRandom()
          return { kind: 'aff', aff: quotient }
        }
        const d = (y as Cell & { kind: 'aff' }).aff.offset
        return { kind: 'aff', aff: scaleAff(affOf(x as Cell), 1 / d) }
      })
    case 'sum':
    case 'mean':
      return reduce(cx, a, typeof params['dim'] === 'string' ? params['dim'] : undefined, op === 'mean')
    case 'at':
      return gather(cx, mode, a, b)
    case 'where':
      return whereGate(cx, mode, args as [Sym, Sym, Sym])
    case 'gt':
    case 'lt':
    case 'eq':
      return cx.zip([a, b], ([x, y]) => compare(cx, op, x as Cell, y as Cell))
    case 'maximum':
    case 'minimum':
      return cx.zip([a, b], ([x, y], i) => ({
        kind: 'aff',
        aff: extremum(cx, op === 'maximum' ? 'max' : 'min', affOf(x as Cell), affOf(y as Cell), i),
      }))
    case 'matmul':
      return matmul(cx, a, b)
    case 'stickBreaking':
      return stickBreaking(cx, a)
    case 'pick':
      return pick(cx, mode, a, b)
    case 'exp':
    case 'square':
    case 'tanh':
    case 'softplus':
    case 'abs':
    case 'sigmoid':
      return { dims: a.dims, cells: a.cells.map((c, i) => ({ kind: 'aff', aff: applyAff(cx, op, affOf(c), i) })) }
    case 'sqrt':
      // The square root of a Gaussian value, or of a known multiple of a gamma variable plus a known offset (a precision).
      return {
        dims: a.dims,
        cells: a.cells.map((c, i): Cell => {
          const gammaArg = scaledOf(c)
          return {
            kind: 'aff',
            aff: gammaArg === undefined ? applyAff(cx, op, affOf(c), i) : applyGamma(cx, op, gammaArg, i),
          }
        }),
      }
    case 'clip':
      // Known bounds: one more step in the chain of the value's `apply` factor (the Cholesky factor of a random correlation, `sqrt(clip(1 - rho^2, ...))`).
      return cx.zip([a, b, args[2] as Sym], ([x, lo, hi], i) => {
        if (!isConstCell(lo as Cell) || !isConstCell(hi as Cell)) throw operatorOnRandom(op)
        const bound = (c: unknown): number => (c as Cell & { kind: 'aff' }).aff.offset
        return { kind: 'aff', aff: clipAff(cx, affOf(x as Cell), bound(lo), bound(hi), i) }
      })
    default:
      throw operatorOnRandom(op)
  }
}

/**
 * The stick-breaking weights of beta latents along the last dimension: one `sticks` cell for each index of the leading
 * dimensions, holding the beta variables of its breaks (the categorical that reads it makes the `stick-breaking` factor).
 */
export function stickBreaking(cx: Context, a: Sym): Sym {
  const sticks = a.dims.length === 0 ? 0 : cx.size(a.dims[a.dims.length - 1] as string)
  if (sticks === 0) throw stickBreakingNeedsDimension()
  const cells: Cell[] = []
  for (let i = 0; i < a.cells.length; i += sticks) {
    const breaks = a.cells.slice(i, i + sticks)
    if (breaks.some((c) => c.kind !== 'rv' || c.family !== 'beta')) throw breaksNotBeta()
    cells.push({ kind: 'sticks', variables: breaks.map((c) => (c as { readonly variable: string }).variable) })
  }
  return { dims: a.dims.slice(0, -1), cells }
}

export function reduce(cx: Context, a: Sym, dim: string | undefined, mean: boolean): Sym {
  const outDims = dim === undefined ? [] : a.dims.filter((d) => d !== dim)
  const shape = cx.shapeOf(a.dims)
  const outShape = cx.shapeOf(outDims)
  // banned-ok: non-numeric: one symbolic affine form per output cell
  const groups: Aff[] = Array.from({ length: prod(outShape) }, () => constAff(0))
  const flats = a.cells.map((_, i) => {
    const idx = unflatten(i, shape)
    const out = outDims.map((d) => idx[a.dims.indexOf(d)] as number)
    let flat = 0
    outDims.forEach((_, k) => {
      flat = flat * (outShape[k] as number) + (out[k] as number)
    })
    return flat
  })
  const counts = toFlat(bincount(flats, prod(outShape)))
  flats.forEach((flat, i) => {
    groups[flat] = combine(groups[flat] as Aff, affOf(a.cells[i] as Cell), 1)
  })
  return {
    dims: outDims,
    cells: groups.map((g, k) => ({ kind: 'aff', aff: mean ? scaleAff(g, 1 / (counts[k] as number)) : g })),
  }
}

export function matmul(cx: Context, a: Sym, b: Sym): Sym {
  const inner = a.dims[a.dims.length - 1]
  if (inner === undefined || inner !== b.dims[0] || b.dims.length > 2) throw matmulShapes(a.dims, b.dims)
  const left = a.dims.slice(0, -1)
  const right = b.dims.slice(1)
  const dims = [...left, ...right]
  const shape = cx.shapeOf(dims)
  const d = cx.size(inner)
  const cells: Cell[] = []
  for (let i = 0; i < prod(shape); i++) {
    const at = unflatten(i, shape)
    let acc = constAff(0)
    for (let j = 0; j < d; j++) {
      const x = cx.cellAt({ dims: a.dims, cells: a.cells }, a.dims, [...at.slice(0, left.length), j])
      const y = cx.cellAt({ dims: b.dims, cells: b.cells }, b.dims, [j, ...at.slice(left.length)])
      acc = combine(acc, mulAff(cx, affOf(x), affOf(y), i), 1)
    }
    cells.push({ kind: 'aff', aff: acc })
  }
  return { dims, cells }
}
