import { bernoulliEta, dirichletPrior, discretePrior } from '../../../../message-passing/index.ts'
import { asTensor, toFlat } from '../../../../tensor/index.ts'
import type { NamedValue } from '../../operators.ts'
import type { DistSpec, ExprSpec, ModelNode } from '../../spec.ts'
import { discreteRowGate } from '../_builder.ts'
import { type Cell, prod, unflatten } from '../_cells.ts'
import type { Context } from '../_context.ts'
import { randomTransitionRow, transitionRow } from '../_idioms.ts'
import { constParam, discreteParents, eventVector, rvName } from '../_params.ts'
import {
  concentrationMissing,
  concentrationRandom,
  discreteParentsOverPlate,
  logitsRandom,
  previousStatesVersusRows,
  probabilitiesNotConjugate,
  probabilityMissing,
  probabilityNotConjugate,
  probabilityOutsideUnit,
  probsOrLogitsMissing,
  rowNotChosenByDiscrete,
  rowsNotDirichlet,
  transitionMatrixSize,
} from '../_reasons.ts'
import { Refusal } from '../_refusal.ts'
import { evalSym } from '../_symbolic/_affine.ts'

export function latentDirichlet(cx: Context, node: ModelNode & { readonly kind: 'latent' }): void {
  const e = node.dist.params['concentration']
  if (e === undefined) throw concentrationMissing(node.name)
  let nv: NamedValue
  try {
    nv = cx.evalNamed(e)
  } catch (err) {
    if (err instanceof Refusal) throw concentrationRandom(node.name)
    throw err
  }
  const shape = cx.shapeOf(node.dims)
  const cells: Cell[] = []
  const vars: string[] = []
  for (let i = 0; i < prod(shape); i++) {
    const alpha = eventVector(cx, nv, node.dims, unflatten(i, shape))
    const name = cx.g.addVariable({
      name: cx.elementName(node.name, node.dims, i),
      family: { kind: 'dirichlet', size: alpha.length },
    })
    cx.g.addFactor(dirichletPrior(name, alpha), name)
    cells.push({ kind: 'rv', variable: name, family: 'dirichlet' })
    vars.push(name)
  }
  cx.finishLatent(node, 'dirichlet', cells, vars)
}

/** The probability of 1 of a Bernoulli node as a function of its discrete parents, by enumeration, as a table over (parents..., node). */
export function bernoulliTable(
  cx: Context,
  node: ModelNode & { readonly dist: DistSpec },
  parents: readonly string[],
): number[] {
  const e = node.dist.params['probs'] ?? node.dist.params['logits']
  if (e === undefined) throw probabilityMissing(node.name)
  const logits = node.dist.params['probs'] === undefined
  const values: number[] = []
  for (let a = 0; a < 2 ** parents.length; a++) {
    const override = new Map<string, NamedValue>()
    parents.forEach((p, k) => {
      const bit = (a >> (parents.length - 1 - k)) & 1
      override.set(p, { value: asTensor(bit, 'float64'), dims: [] })
    })
    const raw = toFlat(cx.evalNamed(e, override).value)[0] as number
    const p1 = logits ? 1 / (1 + Math.exp(-raw)) : raw
    if (!(p1 >= 0 && p1 <= 1)) throw probabilityOutsideUnit(node.name, p1)
    values.push(1 - p1, p1)
  }
  return values
}

/**
 * A categorical latent: a `discrete` variable per element with a known probability vector (a `prior` factor), or with a
 * Dirichlet latent as its probabilities (the conjugate `discrete` factor, a mixture weight). Its role in a model is as
 * the gate of `at(components, z)` or the index of a conjugate observation.
 */
export function latentCategorical(cx: Context, node: ModelNode & { readonly kind: 'latent' }): void {
  const shape = cx.shapeOf(node.dims)
  const n = prod(shape)
  const probsExpr = node.dist.params['probs']
  const logitsExpr = node.dist.params['logits']
  const expr = probsExpr ?? logitsExpr
  if (expr === undefined) throw probsOrLogitsMissing(node.name)
  const cells: Cell[] = []
  const vars: string[] = []
  const chain = logitsExpr === undefined && !cx.isConst(expr) ? transitionRow(cx.idiom, expr) : undefined
  if (chain !== undefined) {
    // The probabilities are a known row of a matrix chosen by another discrete variable (a Markov chain's transition): a table over the two.
    const parents = cx.over(evalSym(cx, chain.index), node.dims)
    const matrix = toFlat(cx.evalNamed(chain.matrix).value)
    for (const [i, p] of parents.entries()) {
      if (p.kind !== 'rv' || (p.family !== 'discrete' && p.family !== 'bernoulli'))
        throw rowNotChosenByDiscrete(node.name)
      const from = p.family === 'bernoulli' ? 2 : cx.g.sizeOf(p.variable)
      const to = matrix.length / from
      if (!Number.isInteger(to) || to < 1) throw transitionMatrixSize(node.name, matrix.length, from)
      const name = cx.g.addVariable({
        name: cx.elementName(node.name, node.dims, i),
        family: { kind: 'discrete', size: to },
      })
      cx.g.need('table')
      cx.g.addFactor(
        { operator: 'table', args: [p.variable, name], params: { states: [from, to], values: Array.from(matrix) } },
        name,
      )
      cells.push({ kind: 'rv', variable: name, family: 'discrete' })
      vars.push(name)
    }
  } else if (!cx.isConst(expr) && randomTransitionRow(cx.idiom, expr) !== undefined) {
    // The rows of the matrix are Dirichlet latents (a transition matrix to learn), chosen by another discrete variable: one
    // `gate` per element, whose state is the previous state and whose branch `j` is the conjugate `discrete` factor of this
    // state on row `j`. Each row then hears only the transitions out of its state, weighted by the belief that the chain was there.
    const chosen = randomTransitionRow(cx.idiom, expr) as { readonly matrix: ExprSpec; readonly index: ExprSpec }
    const matrix = evalSym(cx, chosen.matrix)
    const rows = matrix.cells
    const parents = cx.over(evalSym(cx, chosen.index), node.dims)
    if (matrix.dims.length !== 1 || rows.some((r) => r.kind !== 'rv' || r.family !== 'dirichlet'))
      throw rowsNotDirichlet(node.name)
    const rowVars = rows.map((r) => (r as { readonly variable: string }).variable)
    const size = cx.g.sizeOf(rowVars[0] as string)
    for (const [i, p] of parents.entries()) {
      if (p.kind !== 'rv' || (p.family !== 'discrete' && p.family !== 'bernoulli'))
        throw rowNotChosenByDiscrete(node.name)
      const from = p.family === 'bernoulli' ? 2 : cx.g.sizeOf(p.variable)
      if (from !== rowVars.length) throw previousStatesVersusRows(node.name, from, rowVars.length)
      const name = cx.g.addVariable({
        name: cx.elementName(node.name, node.dims, i),
        family: { kind: 'discrete', size },
      })
      cx.g.need('gate')
      cx.g.need('discrete')
      cx.g.addFactor(discreteRowGate(p.variable, name, rowVars), name)
      cells.push({ kind: 'rv', variable: name, family: 'discrete' })
      vars.push(name)
    }
  } else if (!cx.isConst(expr)) {
    if (logitsExpr !== undefined) throw logitsRandom(node.name)
    const over = cx.over(evalSym(cx, expr), node.dims)
    for (const [i, c] of over.entries()) {
      if (c.kind === 'sticks') {
        // The weights are the stick-breaking of beta breaks: a `stick-breaking` factor on the assignment and the breaks.
        const name = cx.g.addVariable({
          name: cx.elementName(node.name, node.dims, i),
          family: { kind: 'discrete', size: c.variables.length + 1 },
        })
        cx.g.need('stick-breaking')
        cx.g.addFactor(
          { operator: 'stick-breaking', args: [name, ...c.variables], params: { sticks: c.variables.length } },
          name,
        )
        cells.push({ kind: 'rv', variable: name, family: 'discrete' })
        vars.push(name)
        continue
      }
      if (c.kind !== 'rv' || c.family !== 'dirichlet') throw probabilitiesNotConjugate(node.name)
      const size = cx.g.sizeOf(c.variable)
      const name = cx.g.addVariable({
        name: cx.elementName(node.name, node.dims, i),
        family: { kind: 'discrete', size },
      })
      cx.g.addFactor({ operator: 'discrete', args: [name, c.variable] }, name)
      cells.push({ kind: 'rv', variable: name, family: 'discrete' })
      vars.push(name)
    }
  } else {
    const nv = cx.evalNamed(expr)
    for (let i = 0; i < n; i++) {
      const raw = eventVector(cx, nv, node.dims, unflatten(i, shape))
      const probs =
        logitsExpr === undefined
          ? raw
          : (() => {
              const top = Math.max(...raw)
              const e = raw.map((v) => Math.exp(v - top))
              const t = e.reduce((a, b) => a + b, 0)
              return e.map((v) => v / t)
            })()
      const name = cx.g.addVariable({
        name: cx.elementName(node.name, node.dims, i),
        family: { kind: 'discrete', size: probs.length },
      })
      cx.g.addFactor(discretePrior(name, probs), name)
      cells.push({ kind: 'rv', variable: name, family: 'discrete' })
      vars.push(name)
    }
  }
  cx.finishLatent(node, 'discrete', cells, vars)
}

export function latentDiscrete(cx: Context, node: ModelNode & { readonly kind: 'latent' }): void {
  if (node.dist.family === 'categorical') {
    latentCategorical(cx, node)
    return
  }
  const parents = discreteParents(cx, node)
  if (node.dims.length > 0 && parents.length > 0) throw discreteParentsOverPlate(node.name, node.dims)
  const shape = cx.shapeOf(node.dims)
  const n = prod(shape)
  const cells: Cell[] = []
  const vars: string[] = []
  const probsExpr = node.dist.params['probs']
  if (parents.length === 0 && probsExpr !== undefined && !cx.isConst(probsExpr)) {
    // A Bernoulli whose probability is a beta latent (an unknown mixing weight, a gate's prior): the conjugate `bernoulli` factor,
    // whose message to the probability is the moment-matched mixture of the two conjugate updates.
    cx.g.need('bernoulli')
    const over = cx.over(evalSym(cx, probsExpr), node.dims)
    for (const [i, c] of over.entries()) {
      if (c.kind !== 'rv' || c.family !== 'beta') throw probabilityNotConjugate(node.name)
      const name = cx.g.addVariable({
        name: cx.elementName(node.name, node.dims, i),
        family: { kind: 'bernoulli' },
      })
      cx.g.addFactor({ operator: 'bernoulli', args: [name, c.variable] }, name)
      cells.push({ kind: 'rv', variable: name, family: 'bernoulli' })
      vars.push(name)
    }
  } else if (parents.length === 0) {
    const probs = constParam(cx, node, node.dist.params['probs'] === undefined ? 'logits' : 'probs', 'the probability')
    const p = node.dist.params['probs'] === undefined ? probs.map((l) => 1 / (1 + Math.exp(-l))) : probs
    for (let i = 0; i < n; i++) {
      const name = cx.g.addVariable({
        name: cx.elementName(node.name, node.dims, i),
        family: { kind: 'bernoulli' },
      })
      cx.g.addFactor({ operator: 'prior', args: [name], params: { eta: [...bernoulliEta(p[i] as number)] } }, name)
      cells.push({ kind: 'rv', variable: name, family: 'bernoulli' })
      vars.push(name)
    }
  } else {
    const name = cx.g.addVariable({ name: node.name, family: { kind: 'bernoulli' } })
    const pv = parents.map((p) => rvName(cx, p))
    cx.g.addFactor(
      {
        operator: 'table',
        args: [...pv, name],
        params: { states: new Array<number>(parents.length + 1).fill(2), values: bernoulliTable(cx, node, parents) },
      },
      name,
    )
    cells.push({ kind: 'rv', variable: name, family: 'bernoulli' })
    vars.push(name)
  }
  cx.finishLatent(node, 'bernoulli', cells, vars)
}
