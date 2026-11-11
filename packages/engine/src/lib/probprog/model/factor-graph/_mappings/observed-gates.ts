import { bernoulliEta, likelihoods } from '../../../../message-passing/index.ts'
import type { ExprSpec, ModelNode } from '../../spec.ts'
import { discreteRowGate, GateBuilder, precisionBranch } from '../_builder.ts'
import { type Aff, affOf, type Cell, combine, constAff, isConstCell, type Precision } from '../_cells.ts'
import type { Context } from '../_context.ts'
import { expArgument, mentionsDiscrete, precisionExpr, randomTransitionRow, stripClip, unwrap } from '../_idioms.ts'
import { NOT_MINE, type NotMine, probe } from '../_mapping.ts'
import { cellsOf, paramCell, rvName } from '../_params.ts'
import { Refusal } from '../_refusal.ts'
import { evalSym } from '../_symbolic/_affine.ts'
import { gate } from '../_symbolic/_gates.ts'
import { bernoulliTable } from './latent-discrete.ts'

/**
 * A Gaussian observation of a value that is a known function of a discrete variable (a reading moved a period early or late by a
 * copy index): `N(v; m, s)` is `N(0; m - v, s)`, so the observed value moves across to the mean, which then is a gate over the
 * shifted forms. Undefined when the observed value is known, or the family is not the normal.
 */
export function valueMovedToMean(
  cx: Context,
  node: ModelNode & { readonly kind: 'observed' },
): (ModelNode & { readonly kind: 'observed' }) | undefined {
  const loc = node.dist.params['loc']
  if (node.dist.family !== 'normal' || loc === undefined || cx.isConst(node.value)) return undefined
  if (!mentionsDiscrete(cx.latents, node.value)) return undefined
  const fused = gatedMean(cx, node, loc)
  if (fused !== undefined) return { ...node, value: { kind: 'const', value: 0 } }
  return {
    ...node,
    dist: { ...node.dist, params: { ...node.dist.params, loc: { kind: 'op', op: 'sub', args: [loc, node.value] } } },
    value: { kind: 'const', value: 0 },
  }
}

/**
 * The mean `at(table, z)` of an observation whose value is a known function of the same discrete variable `z`: one gate over the
 * table's entries each moved by what the value is in that state (`table[k] - value[k]`), so the observed value becomes zero and
 * the per-state shifts sit in the one gate (a gate of the shifted forms, not a gate of the mean followed by another of the
 * value). Stores the cells of the mean for the observation to read; false when the mean is not of that form.
 */
export function gatedMean(cx: Context, node: ModelNode & { readonly kind: 'observed' }, loc: ExprSpec): boolean {
  const e = unwrap(cx.nodes, loc)
  if (e.kind !== 'op' || e.op !== 'at') return false
  const [first, index] = e.args as [ExprSpec, ExprSpec]
  const table = evalSym(cx, first)
  if (table.dims.length !== 1) return false
  const values = cx.over(evalSym(cx, node.value), node.dims)
  const picks = cx.over(evalSym(cx, index), node.dims)
  const cells: Cell[] = []
  for (const [i, v] of values.entries()) {
    const pick = picks[i] as Cell
    if (v.kind !== 'lookup' || v.items !== undefined || pick.kind !== 'rv' || pick.variable !== v.variable) return false
    if (table.cells.length !== v.values.length || !table.cells.every((c) => c.kind === 'aff')) return false
    const shifted = table.cells.map(
      (c, k): Cell => ({
        kind: 'aff',
        aff: combine(affOf(c), constAff(v.values[k] as number), -1),
      }),
    )
    cells.push(gate(cx, 'plain', shifted, v.variable))
  }
  cx.cells.set(`${node.name}|loc`, cells)
  return true
}

/**
 * A Poisson count whose rate is `k_z exp(L)`: `L` an affine form of Gaussian variables and `k_z` a known exposure chosen by a
 * discrete variable `z` (zero for a state that is a structural zero: zero inflation). The state of `z` selects which
 * likelihood holds, so it is a `gate` whose branches are the Poisson likelihood of `L` (a state with exposure zero is the
 * constant one, a count of zero) with a `table` that rules out the states in which a positive count has probability zero; when
 * `L` is known the gate is a table over `z`. The weights of `z` may be known or a Dirichlet latent, as for any discrete variable.
 */
export function observedPoissonGate(
  cx: Context,
  node: ModelNode & { readonly kind: 'observed' },
  y: number,
  i: number,
  index: number | undefined,
  inner: ExprSpec,
  exposure: ExprSpec,
): null {
  const cell = cellsOf(cx, node, 'exposure', () => evalSym(cx, exposure))[i] as Cell
  if (cell.kind !== 'lookup' || cell.items !== undefined)
    throw new Refusal(
      'operator',
      `the exposure of '${node.name}' is random: the poisson likelihood has a message rule for a known exposure, or a known value chosen by a discrete variable (a gate)`,
    )
  const z = cell.variable
  const exposures = cell.values
  if (!exposures.every((e) => e >= 0 && Number.isFinite(e)))
    throw new Refusal('data', `the exposure of '${node.name}' must be non-negative`)
  const why = likelihoods()['poisson-log']?.check?.(y, { likelihood: 'poisson-log', y, exposure: 1 })
  if (why !== undefined) throw new Refusal('data', `'${node.name}' (poisson-log): ${why}`)
  const predictor = cellsOf(cx, node, 'predictor', () => evalSym(cx, inner))[i] as Cell
  const id = cx.elementName(node.name, node.dims, i)
  const impossible = exposures.map((e) => e === 0 && y > 0)
  const logFactorial = Array.from({ length: y }, (_, k) => Math.log(k + 1)).reduce((a, b) => a + b, 0)
  if (isConstCell(predictor)) {
    // Nothing random but the state: the likelihood of the count under each state.
    const pmf = (e: number): number => {
      const rate = e * Math.exp(predictor.aff.offset)
      return rate === 0 ? (y === 0 ? 1 : 0) : Math.exp(-rate + y * Math.log(rate) - logFactorial)
    }
    cx.g.addFactor(
      { operator: 'table', args: [z], params: { states: [exposures.length], values: exposures.map(pmf) } },
      id,
    )
    return null
  }
  const v = cx.g.materialise(affOf(predictor), node.name, index) as string
  if (impossible.some(Boolean))
    cx.g.addFactor(
      {
        operator: 'table',
        args: [z],
        params: { states: [exposures.length], values: impossible.map((bad) => (bad ? 0 : 1)) },
      },
      `${id}|support`,
    )
  const gate = new GateBuilder(z, [v])
  for (const e of exposures)
    gate.add(
      e === 0 ? {} : { operator: 'likelihood-site', args: [0], params: { likelihood: 'poisson-log', y, exposure: e } },
    )
  cx.g.addFactor(gate.factor(), id)
  return null
}

/**
 * A Bernoulli variable no model node names, the switch that a summed-out mixture (`normalOutliers`, `normalSpikeSlab`) with a
 * random share or random precisions is read through: the mixture is the gate of its components over this switch, which is the
 * same graph as the explicit-switch form (`z ~ bernoulli(share)`, `where(z, ...)`) with the switch left out of the posterior.
 * A beta share is the conjugate `bernoulli` factor, a known one a prior on the switch.
 */
export function hiddenSwitch(cx: Context, node: ModelNode, probs: Cell, index: number | undefined): string {
  if (probs.kind === 'rv' && probs.family === 'beta') {
    const name = cx.g.addVariable({ name: cx.g.auxName(node.name, index), family: { kind: 'bernoulli' } })
    cx.g.addFactor({ operator: 'bernoulli', args: [name, probs.variable] }, name)
    return name
  }
  if (!isConstCell(probs))
    throw new Refusal(
      'operator',
      `the share of '${node.name}' must be known or a beta latent (the conjugate 'bernoulli' operator on the switch of a mixture)`,
    )
  const p = probs.aff.offset
  if (!(p > 0 && p < 1))
    throw new Refusal('data', `the share of '${node.name}' is ${p}: it must be strictly between 0 and 1`)
  const name = cx.g.addVariable({ name: cx.g.auxName(node.name, index), family: { kind: 'bernoulli' } })
  cx.g.addFactor({ operator: 'prior', args: [name], params: { eta: [...bernoulliEta(p)] } }, name)
  return name
}

/** The location and scale cells of an observed normal element when one of them is a lookup by a discrete variable; absent otherwise. */
export function switchedCells(
  cx: Context,
  node: ModelNode & { readonly kind: 'observed' },
  i: number,
): { readonly loc: Cell; readonly scale: Cell } | undefined {
  const locExpr = node.dist.params['loc']
  const scaleExpr = node.dist.params['scale']
  const discrete = (e: ExprSpec | undefined): boolean =>
    e !== undefined && !cx.isConst(e) && mentionsDiscrete(cx.latents, e)
  if (!discrete(locExpr) && !discrete(scaleExpr)) return undefined
  // A precision chosen by a discrete variable (`1 / sqrt(where(z, tau_b, tau_a))`, `1 / sqrt(at(tau, z))`): the mean and the
  // precision are read as lookups of that variable, so the observation is one gate of `gaussian-gamma-precision` branches.
  const precision = scaleExpr === undefined ? undefined : precisionExpr(cx.idiom, scaleExpr)
  if (precision !== undefined && discrete(precision)) {
    const loc = paramCell(cx, node, 'loc', i, 'lookup')
    const lookedUp = cellsOf(cx, node, 'precisionLookup', () => evalSym(cx, precision, 'lookup'))[i] as Cell
    if (lookedUp.kind === 'lookup' && lookedUp.precisions !== undefined) return { loc, scale: lookedUp }
  }
  const loc = paramCell(cx, node, 'loc', i)
  // The precision idiom `1 / sqrt(tau)` where `tau` is a gamma variable, or one chosen among gamma variables by a discrete variable.
  const randomPrecision = (): boolean => {
    const p = scaleExpr === undefined ? undefined : precisionExpr(cx.idiom, scaleExpr)
    if (p === undefined) return false
    const c = probe(cx, () => cellsOf(cx, node, 'precisionProbe', () => evalSym(cx, p))[i] as Cell)
    return c !== NOT_MINE && (c.kind === 'scaled' || (c.kind === 'rv' && c.family === 'gamma'))
  }
  // A mean that mentions a discrete variable but is a random Gaussian after its gates (a `select`), with a scale that does not
  // (a precision idiom `1 / sqrt(tau)`), is an ordinary observation: the scale is not evaluated as a lookup.
  if (loc.kind !== 'lookup' && (!discrete(scaleExpr) || randomPrecision())) return undefined
  const scale = paramCell(cx, node, 'scale', i)
  return loc.kind === 'lookup' || scale.kind === 'lookup' ? { loc, scale } : undefined
}

/**
 * An observed Gaussian whose location and (or) scale is a lookup by one discrete variable `z`: when the location is known in
 * every state, a table over `z` (the likelihood of the value under each state: a hidden Markov model's emission); when it is
 * random in some states, a `gate` of one Gaussian likelihood per state, over the random variables of the branches (the clutter
 * problem, a mixture with a known component, a change point).
 */
export function observedSwitch(
  cx: Context,
  node: ModelNode & { readonly kind: 'observed' },
  name: string,
  y: number,
  index: number | undefined,
  loc: Cell,
  scale: Cell,
): null {
  const looks = [loc, scale].filter((c): c is Cell & { kind: 'lookup' } => c.kind === 'lookup')
  const z = (looks[0] as Cell & { kind: 'lookup' }).variable
  if (looks.some((l) => l.variable !== z))
    throw new Refusal(
      'operator',
      `the location and the scale of '${node.name}' are chosen by different discrete variables`,
    )
  const states = (looks[0] as Cell & { kind: 'lookup' }).values.length
  if (looks.some((l) => l.values.length !== states))
    throw new Refusal('operator', `the lookups of '${node.name}' have different numbers of states`)
  const precisions = scale.kind === 'lookup' ? scale.precisions : undefined
  if (scale.kind === 'lookup' && scale.items !== undefined)
    throw new Refusal(
      'operator',
      `the scale of '${node.name}' is random in some states of a discrete variable: it must be known`,
    )
  if (scale.kind !== 'lookup' && !isConstCell(scale))
    throw new Refusal('operator', `the scale of '${node.name}' is not known`)
  // With a precision lookup, `values` are the known precisions of the states that have no gamma variable, not scales.
  const sdAt = (k: number): number =>
    precisions !== undefined
      ? 1 / Math.sqrt((scale as Cell & { kind: 'lookup' }).values[k] as number)
      : scale.kind === 'lookup'
        ? (scale.values[k] as number)
        : (scale as Cell & { kind: 'aff' }).aff.offset
  for (let k = 0; k < states; k++) {
    if (precisions?.[k] !== undefined) {
      const p = precisions[k] as Precision
      if (!(p.scale >= 0 && p.offset >= 0 && p.scale + p.offset > 0))
        throw new Refusal('data', `the precision of '${node.name}' must be a non-negative multiple of a gamma variable`)
      continue
    }
    if (!(sdAt(k) > 0)) throw new Refusal('data', `the scale of '${node.name}' is ${sdAt(k)}: it must be positive`)
  }
  const locAt = (k: number): Aff | number => {
    if (loc.kind === 'lookup') return loc.items?.[k] ?? (loc.values[k] as number)
    return affOf(loc)
  }
  const known =
    precisions === undefined &&
    Array.from({ length: states }, (_, k) => locAt(k)).every((m) => typeof m === 'number' || m.terms.size === 0)
  const pdf = (m: number, sd: number): number => Math.exp(-0.5 * ((y - m) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI))
  const value = (m: Aff | number): number => (typeof m === 'number' ? m : m.offset)
  if (known) {
    cx.g.need('table')
    cx.g.addFactor(
      {
        operator: 'table',
        args: [z],
        params: { states: [states], values: Array.from({ length: states }, (_, k) => pdf(value(locAt(k)), sdAt(k))) },
      },
      name,
    )
    return null
  }
  cx.g.need('gate')
  const gate = new GateBuilder(z)
  for (let k = 0; k < states; k++) {
    const m = locAt(k)
    let mean: number | { readonly const: number }
    if (typeof m === 'number' || m.terms.size === 0) mean = { const: value(m) }
    else mean = gate.slot(cx.g.materialise(m, node.name, index) as string)
    const gamma = precisions?.[k]
    if (gamma !== undefined) {
      const at = gate.slot(gamma.variable)
      cx.g.need('gaussian-gamma-precision')
      gate.add(precisionBranch([{ const: y }, mean, at], gamma))
    } else
      gate.add({
        operator: 'gaussian',
        args: [{ const: y }, typeof mean === 'number' ? mean : mean],
        params: { precision: 1 / sdAt(k) ** 2 },
      })
  }
  cx.g.addFactor(gate.factor(), name)
  return null
}

/**
 * An observed Gaussian whose scale is `exp(s)` with `s` chosen by a discrete variable `z` among Gaussian forms of its own (a spread
 * per component of a mixture, `exp(at(logSd, z))`), whose mean may be chosen by the same `z` among random ones: ONE `gate` on `z`
 * whose branch `k` is the `gaussian-scale` factor of the observation on the mean and the log scale of state `k`, so the
 * component that explains the reading is the one whose mean and spread both move (a `select` of each would let EP lose which
 * component the pair came from, and a spread per component run off). `NOT_MINE` when the scale is not of that form (the other
 * routes are tried). The observation is the gate, so the element has no variable of its own (the result is `null`).
 */
export function observedSwitchLogScale(
  cx: Context,
  node: ModelNode & { readonly kind: 'observed' },
  y: number,
  i: number,
  index: number | undefined,
): null | NotMine {
  const scaleExpr = node.dist.params['scale']
  if (scaleExpr === undefined || cx.isConst(scaleExpr)) return NOT_MINE
  const logArgument = expArgument(stripClip(cx.idiom, scaleExpr))
  if (logArgument === undefined || !mentionsDiscrete(cx.latents, logArgument)) return NOT_MINE
  const cells = probe(cx, () => ({
    loc: paramCell(cx, node, 'loc', i, 'lookup'),
    logScale: cellsOf(cx, node, 'logScaleLookup', () => evalSym(cx, logArgument, 'lookup'))[i] as Cell,
  }))
  if (cells === NOT_MINE) return NOT_MINE
  const { loc, logScale } = cells
  if (logScale.kind !== 'lookup' || logScale.precisions !== undefined) return NOT_MINE
  const z = logScale.variable
  const states = logScale.values.length
  if (loc.kind === 'lookup' && (loc.variable !== z || loc.values.length !== states || loc.precisions !== undefined))
    throw new Refusal(
      'operator',
      `the mean and the spread of '${node.name}' are chosen by different discrete variables`,
    )
  if (loc.kind !== 'lookup' && loc.kind !== 'aff')
    throw new Refusal('operator', `the mean of '${node.name}' is not a Gaussian form or a lookup`)
  const locAt = (k: number): Aff | number => {
    if (loc.kind === 'lookup') return loc.items?.[k] ?? (loc.values[k] as number)
    return affOf(loc)
  }
  const gate = new GateBuilder(z)
  const argOf = (form: Aff | number | undefined, fallback: number): number | { readonly const: number } => {
    if (form === undefined || typeof form === 'number' || form.terms.size === 0)
      return { const: form === undefined ? fallback : typeof form === 'number' ? form : form.offset }
    return gate.slot(cx.g.materialise(form, node.name, index) as string)
  }
  cx.g.need('gate')
  cx.g.need('gaussian-scale')
  for (let k = 0; k < states; k++)
    gate.add({
      operator: 'gaussian-scale',
      args: [{ const: y }, argOf(locAt(k), 0), argOf(logScale.items?.[k], logScale.values[k] as number)],
      params: {},
    })
  cx.g.addFactor(gate.factor(), cx.elementName(node.name, node.dims, i))
  return null
}

/**
 * An observed categorical whose probabilities are the row of a matrix of Dirichlet latents chosen by a discrete latent
 * (`categorical({ probs: phi.at(z) })`, the words of a topic model: `phi_k` the word distribution of topic `k`, `z` the topic of
 * the token). It is one `gate` on the index whose branch `k` is the conjugate `discrete` factor of the observed outcome on row
 * `k`: each row hears the tokens, weighted by the belief that the token chose it, and the index hears how well each row
 * explains the outcome. The same gate as an unknown transition row, with an observed outcome.
 */
export function observedRowGate(
  cx: Context,
  node: ModelNode & { readonly kind: 'observed' },
  name: string,
  y: number,
  i: number,
  expr: ExprSpec,
): string | null {
  const chosen = randomTransitionRow(cx.idiom, expr) as { readonly matrix: ExprSpec; readonly index: ExprSpec }
  const matrix = evalSym(cx, chosen.matrix)
  if (matrix.dims.length !== 1 || matrix.cells.some((r) => r.kind !== 'rv' || r.family !== 'dirichlet'))
    throw new Refusal(
      'operator',
      `the rows of the matrix behind '${node.name}' must be one Dirichlet latent each (a latent over the dimension of the states)`,
    )
  const rowVars = matrix.cells.map((r) => (r as { readonly variable: string }).variable)
  const size = cx.g.sizeOf(rowVars[0] as string)
  const parent = cx.over(evalSym(cx, chosen.index), node.dims)[i] as Cell
  if (parent.kind !== 'rv' || (parent.family !== 'discrete' && parent.family !== 'bernoulli'))
    throw new Refusal('operator', `the row of '${node.name}' must be chosen by a discrete latent, not a computed value`)
  const from = parent.family === 'bernoulli' ? 2 : cx.g.sizeOf(parent.variable)
  if (from !== rowVars.length)
    throw new Refusal('data', `'${node.name}' has ${from} choices of row and its matrix ${rowVars.length} rows`)
  if (!(Number.isInteger(y) && y >= 0 && y < size))
    throw new Refusal('data', `'${node.name}' observes ${y}, outside its ${size} categories`)
  cx.g.addVariable({ name, family: { kind: 'discrete', size }, observed: y })
  cx.g.need('gate')
  cx.g.need('discrete')
  cx.g.addFactor(discreteRowGate(parent.variable, name, rowVars), name)
  return name
}

/**
 * An observed Bernoulli whose probability depends on discrete latents (a node of a network of discrete variables): one `table` factor
 * over the parents and the node, the probability of 1 for each state of the parents (`bernoulliTable`). A node over a plate has no such
 * table: the tables over plated nodes are refused.
 */
export function observedBernoulliTable(
  cx: Context,
  node: ModelNode & { readonly kind: 'observed' },
  values: readonly number[],
  parents: readonly string[],
): void {
  if (node.dims.length > 0)
    throw new Refusal(
      'structure',
      `'${node.name}' is a Bernoulli over [${node.dims}] with discrete parents: tables over plated nodes are not expanded yet`,
    )
  const name = cx.g.addVariable({ name: node.name, family: { kind: 'bernoulli' }, observed: values[0] as number })
  const pv = parents.map((p) => rvName(cx, p))
  cx.g.addFactor(
    {
      operator: 'table',
      args: [...pv, name],
      params: { states: new Array<number>(pv.length + 1).fill(2), values: bernoulliTable(cx, node, parents) },
    },
    name,
  )
  cx.layout(node, 'observed', 'bernoulli', 'scalar', [name], false)
}
