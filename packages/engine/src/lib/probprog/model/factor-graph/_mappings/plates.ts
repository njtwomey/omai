import type { FactorSpec } from '../../../../message-passing/index.ts'
import { matmul, mul, scatterAdd, sub, sum, take, tensor, toFlat, transpose } from '../../../../tensor/index.ts'
import type { ExprSpec, ModelNode } from '../../spec.ts'
import { type Atom, type Cell, prod } from '../_cells.ts'
import type { Context } from '../_context.ts'
import { unwrap } from '../_idioms.ts'
import { NOT_MINE, type NotMine } from '../_mapping.ts'
import { cellsOf, paramSym } from '../_params.ts'
import { evalSym } from '../_symbolic/_affine.ts'

/**
 * The rows of an observed node as a design matrix on one packed weight vector, when every row is `a . w` (plus a known
 * offset where `offset` is allowed) of the same vector `w`: the vector's name, its size, and the row (`undefined` for a row
 * with no random part, which says nothing about `w`). Absent when the rows are not of that shape.
 */
export function designRows(
  cx: Context,
  cells: readonly Cell[],
  offset: boolean,
):
  | {
      readonly vector: string
      readonly size: number
      readonly rows: readonly ({ a: number[]; offset: number } | undefined)[]
    }
  | undefined {
  let vector: string | undefined
  let size = 0
  const rows: ({ a: number[]; offset: number } | undefined)[] = []
  for (const c of cells) {
    if (c.kind !== 'aff') return undefined
    const terms = [...c.aff.terms].filter(([, k]) => k !== 0)
    if (terms.length === 0) {
      rows.push(undefined)
      continue
    }
    if (c.aff.offset !== 0 && !offset) return undefined
    const atoms = terms.map(([k]) => cx.g.atoms.get(k) as Atom)
    for (const a of atoms) {
      if (a.kind !== 'comp') return undefined
      if (vector === undefined) {
        vector = a.vector
        size = a.size
      } else if (a.vector !== vector) return undefined
    }
    const a = toFlat(
      scatterAdd(
        [size],
        atoms.map((atom) => (atom as Atom & { kind: 'comp' }).index),
        terms.map(([, coefficient]) => coefficient),
        0,
      ),
    )
    rows.push({ a, offset: c.aff.offset })
  }
  return vector === undefined ? undefined : { vector, size, rows }
}

/**
 * An observed node over a plate as one plated factor on its weight vector, when its rows are i.i.d. draws of one
 * regression on a packed vector `w`: a Gaussian likelihood with one known scale (the exact `linear-regression-plate`, by
 * the sufficient statistics of the rows) or a Bernoulli with a probit or logistic link of `a . w` (`probit-plate`,
 * `logistic-plate`, the design matrix and outcomes as parameters). Anything else, or an engine registry without the
 * plate operators, is not handled here and falls to the expansion (`NOT_MINE`).
 */
export function plate(
  cx: Context,
  node: ModelNode & { readonly kind: 'observed' },
  fam: string,
  values: readonly number[],
): true | NotMine {
  if (fam === 'normal') return plateNormal(cx, node, values)
  if (fam === 'bernoulli') return plateBernoulli(cx, node, values)
  return NOT_MINE
}

/** The indices of the rows a plated factor uses: those with a random part and an observed (not missing) value. */
function usedRows(rows: readonly ({ a: number[]; offset: number } | undefined)[], values: readonly number[]): number[] {
  return rows.flatMap((row, i) => (row === undefined || Number.isNaN(values[i]) ? [] : [i]))
}

export function addPlated(cx: Context, node: ModelNode & { readonly kind: 'observed' }, f: FactorSpec): void {
  cx.g.addLibraryFactor(f, node.name)
  cx.plated.push(node.name)
  // banned-ok: non-numeric: a layout of nulls (the cells of an observed node carry no variable)
  cx.layout(node, 'observed', undefined, 'observed', new Array<null>(prod(cx.shapeOf(node.dims))).fill(null), false)
}

export function plateNormal(
  cx: Context,
  node: ModelNode & { readonly kind: 'observed' },
  values: readonly number[],
): true | NotMine {
  if (!('linear-regression-plate' in cx.g.ops)) return NOT_MINE
  const scaleExpr = node.dist.params['scale']
  if (scaleExpr !== undefined && !cx.isConst(scaleExpr)) return NOT_MINE
  const scales = cx
    .over(paramSym(cx, node.dist, 'scale'), node.dims)
    .map((c) => (c as Cell & { kind: 'aff' }).aff.offset)
  const scale = scales[0] as number
  if (!(scale > 0) || scales.some((v) => v !== scale)) return NOT_MINE
  const locs = cellsOf(cx, node, 'loc', () => paramSym(cx, node.dist, 'loc'))
  const design = designRows(cx, locs, true)
  if (design === undefined) return NOT_MINE
  const used = usedRows(design.rows, values)
  const n = used.length
  if (n === 0) return NOT_MINE
  // The sufficient statistics of the used rows: X'X, X'y and y'y, with the known offsets taken off the outcomes.
  const x = tensor(used.map((i) => (design.rows[i] as { a: number[] }).a))
  const y = sub(take(tensor(values), used), tensor(used.map((i) => (design.rows[i] as { offset: number }).offset)))
  const xtx = toFlat(matmul(transpose(x), x))
  const xty = toFlat(matmul(transpose(x), y))
  const yty = toFlat(sum(mul(y, y)))[0] as number
  addPlated(cx, node, {
    operator: 'linear-regression-plate',
    args: [design.vector],
    params: { xtx, xty, yty, n, precision: 1 / scale ** 2 },
  })
  return true
}

export function plateBernoulli(
  cx: Context,
  node: ModelNode & { readonly kind: 'observed' },
  values: readonly number[],
): true | NotMine {
  const probs = node.dist.params['probs']
  const logits = node.dist.params['logits']
  let link: 'probit' | 'sigmoid' | undefined
  let argument: ExprSpec | undefined
  if (logits !== undefined) {
    link = 'sigmoid'
    argument = logits
  } else if (probs !== undefined) {
    const e = unwrap(cx.nodes, probs)
    if (e.kind === 'op' && (e.op === 'probit' || e.op === 'sigmoid')) {
      link = e.op
      argument = e.args[0] as ExprSpec
    }
  }
  if (link === undefined || argument === undefined) return NOT_MINE
  const operator = link === 'probit' ? 'probit-plate' : 'logistic-plate'
  if (!(operator in cx.g.ops)) return NOT_MINE
  const arg = argument
  const cells = cellsOf(cx, node, `link:${link}`, () => evalSym(cx, arg))
  const design = designRows(cx, cells, false)
  if (design === undefined) return NOT_MINE
  const used = usedRows(design.rows, values)
  if (used.length === 0) return NOT_MINE
  const x = toFlat(tensor(used.map((i) => (design.rows[i] as { a: number[] }).a)))
  const y = toFlat(take(tensor(values), used))
  addPlated(cx, node, { operator, args: [design.vector], params: { X: x, y } })
  return true
}
