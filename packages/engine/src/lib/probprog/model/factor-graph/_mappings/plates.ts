import type { FactorSpec } from '../../../../message-passing/index.ts'
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
    const a = new Array<number>(size).fill(0)
    for (const [k, coefficient] of terms) a[(cx.g.atoms.get(k) as Atom & { kind: 'comp' }).index] = coefficient
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

export function addPlated(cx: Context, node: ModelNode & { readonly kind: 'observed' }, f: FactorSpec): void {
  cx.g.addLibraryFactor(f, node.name)
  cx.plated.push(node.name)
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
  const d = design.size
  const xtx = new Array<number>(d * d).fill(0)
  const xty = new Array<number>(d).fill(0)
  let yty = 0
  let n = 0
  for (const [i, row] of design.rows.entries()) {
    const y = (values[i] as number) - (row?.offset ?? 0)
    if (row === undefined || Number.isNaN(values[i])) continue
    n++
    yty += y * y
    for (let a = 0; a < d; a++) {
      xty[a] = (xty[a] as number) + (row.a[a] as number) * y
      for (let b = 0; b < d; b++)
        xtx[a * d + b] = (xtx[a * d + b] as number) + (row.a[a] as number) * (row.a[b] as number)
    }
  }
  if (n === 0) return NOT_MINE
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
  const x: number[] = []
  const y: number[] = []
  for (const [i, row] of design.rows.entries()) {
    if (row === undefined || Number.isNaN(values[i])) continue
    x.push(...row.a)
    y.push(values[i] as number)
  }
  if (y.length === 0) return NOT_MINE
  addPlated(cx, node, { operator, args: [design.vector], params: { X: x, y } })
  return true
}
