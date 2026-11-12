/**
 * The observations of a latent variable model with random Fourier features (`gplvmFourierModel`): the observed matrix `y[point, output]`
 * whose mean is `sum over feature of cos(x . omega + phase) * W`, found by the layout plan (`_gplvm.ts`), as ONE `gplvm-row` factor per
 * point over the point's coordinates (scalar Gaussian variables) and the weights of each output (the latent mapping of `W` made a Gaussian vector per output).
 * The frequencies, phases, noise and the observed row are parameters of the factor. A node the plan took that turns out not to have the
 * variables the factor needs (a coordinate that is not a plain variable, a weight that is not a coordinate of the vector) is handed back
 * (`NOT_MINE`) and compiles row by row as before.
 */

import { gaussian, gaussianMv } from '../../../../message-passing/index.ts'
import { toFlat } from '../../../../tensor/index.ts'
import type { ExprSpec, ModelNode } from '../../spec.ts'
import type { Atom, Cell } from '../_cells.ts'
import type { Context } from '../_context.ts'
import { NOT_MINE, type NotMine } from '../_mapping.ts'
import { evalNamedValue } from '../_params.ts'

/** The variable an x cell is on its own (coefficient one, no offset), or undefined. */
function plainVariable(cx: Context, cell: Cell | undefined): string | undefined {
  if (cell === undefined || cell.kind !== 'aff' || cell.aff.offset !== 0 || cell.aff.terms.size !== 1) return undefined
  const [key, coefficient] = [...cell.aff.terms][0] as [string, number]
  const atom = cx.g.atoms.get(key) as Atom | undefined
  return atom?.kind === 'var' && coefficient === 1 ? atom.name : undefined
}

/** Whether the cells are, in order, the coordinates `0, 1, ...` of the vector variables of the outputs (`features` each, output by output). */
function coordinatesOf(cx: Context, cells: readonly Cell[], vectors: readonly string[], features: number): boolean {
  return (
    cells.length === vectors.length * features &&
    cells.every((cell, i) => {
      if (cell.kind !== 'aff' || cell.aff.offset !== 0 || cell.aff.terms.size !== 1) return false
      const [key, coefficient] = [...cell.aff.terms][0] as [string, number]
      const atom = cx.g.atoms.get(key) as Atom | undefined
      return (
        atom?.kind === 'comp' &&
        atom.vector === vectors[Math.floor(i / features)] &&
        atom.index === i % features &&
        atom.size === features &&
        coefficient === 1
      )
    })
  )
}

/**
 * The observed matrix of a latent variable model with Fourier features: one `gplvm-row` factor per point on its coordinates and the weight
 * vector. `NOT_MINE` when the variables the plan promised are not as it said.
 */
export function observedGplvm(cx: Context, node: ModelNode & { readonly kind: 'observed' }): NotMine | undefined {
  const plan = cx.plan.gplvm
  if (plan === undefined) return NOT_MINE
  const { points, outputs, features, coords, dims } = plan
  const xSym = cx.syms.get(plan.x)
  const wSym = cx.syms.get(plan.w)
  if (xSym === undefined || wSym === undefined) return NOT_MINE
  const xCells = cx.over(xSym, [dims.point, dims.coord])
  const wCells = cx.over(wSym, [dims.output, dims.feature])
  if (!coordinatesOf(cx, wCells, plan.vectors, features)) return NOT_MINE
  const xVariables = xCells.map((cell) => plainVariable(cx, cell))
  if (xVariables.some((v) => v === undefined)) return NOT_MINE
  // The known numbers of the row: frequencies [coord, feature], phases [feature], the noise and the observations [point, output].
  const named = (e: ExprSpec, over: readonly string[]): number[] =>
    cx.over(cx.fromNamed(cx.evalNamed(e)), over).map((c) => (c as Cell & { kind: 'aff' }).aff.offset)
  const omega = named({ kind: 'ref', name: plan.omega }, [dims.coord, dims.feature])
  const phase =
    plan.phase === undefined
      ? Array.from({ length: features }, () => 0)
      : named({ kind: 'ref', name: plan.phase }, [dims.feature])
  const noise = toFlat(cx.evalNamed(node.dist.params['scale'] as ExprSpec).value)[0] as number
  const y = cx
    .over(cx.fromNamed(evalNamedValue(cx, node)), [dims.point, dims.output])
    .map((c) => (c as Cell & { kind: 'aff' }).aff.offset)
  if (y.some((v) => !Number.isFinite(v))) return NOT_MINE
  const operator = cx.g.need('gplvm-row')
  const families = [
    ...Array.from({ length: coords }, () => gaussian),
    ...Array.from({ length: outputs }, () => gaussianMv(features)),
  ]
  for (let n = 0; n < points; n++) {
    const params = {
      coords,
      features,
      outputs,
      omega: [...omega],
      phase: [...phase],
      noise,
      y: y.slice(n * outputs, (n + 1) * outputs),
    }
    if (operator.check?.(params, families) !== undefined) return NOT_MINE
    cx.g.addFactor(
      {
        operator: 'gplvm-row',
        args: [...(xVariables.slice(n * coords, (n + 1) * coords) as string[]), ...plan.vectors],
        params,
      },
      cx.elementName(node.name, [dims.point], n),
    )
  }
  cx.layout(
    node,
    'observed',
    undefined,
    'observed',
    Array.from({ length: points * outputs }, () => null),
    false,
  )
  return undefined
}
