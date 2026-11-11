/**
 * Gaussian latents: a normal with a known or random mean and scale (non-centred `x = loc + scale z` for a random scale), grouped
 * scalars, the precision idioms (shared, random, one gamma per coefficient), spike-and-slab with and without a switch, the
 * truncation of a normal and the priors with no message family that are projected onto a Gaussian once.
 */

import type { Json } from '../../../../base/index.ts'
import { type FactorSpec, gaussianMvPrior, gaussianPrior } from '../../../../message-passing/index.ts'
import type { ExprSpec, ModelNode } from '../../spec.ts'
import { type GateBranch, GateBuilder, precisionBranch, precisionFactor } from '../_builder.ts'
import { affOf, type Cell, combine, isConstCell, logName, type Precision, prod } from '../_cells.ts'
import type { Context } from '../_context.ts'
import { precisionExpr, stripClip } from '../_idioms.ts'
import { NOT_MINE, probe } from '../_mapping.ts'
import { constParam, paramIsConst, paramSym, randomPrecisions } from '../_params.ts'
import { Refusal } from '../_refusal.ts'
import { evalSym } from '../_symbolic/_affine.ts'
import { mulAff } from '../_symbolic/_functions.ts'
import { hiddenSwitch } from './observed-gates.ts'

/**
 * A prior with no message family (half-normal, half-Cauchy, log-normal, uniform, truncated normal), the exact numeric
 * `prior-site` on a Gaussian variable (a log-normal is the exact Gaussian on its log). A density on the positive half line is
 * carried on its log: a Gaussian variable `l = log tau` with the density of `l` as its site, and the value `tau` is the variable
 * `exp(l)` of an `apply` factor, whose Gaussian belief is the moment match of a log-normal. A density on a bounded or truncated
 * real range is the site on the value itself. The site's message is the moment match against the current cavity at every sweep,
 * so the graph holds the prior itself; the belief of the variable is still one Gaussian.
 */
export function latentProjected(cx: Context, node: ModelNode & { readonly kind: 'latent' }): void {
  const fam = node.dist.family
  const n = prod(cx.shapeOf(node.dims))
  const names: Readonly<Record<string, readonly string[]>> = {
    halfNormal: ['scale'],
    halfCauchy: ['scale'],
    logNormal: ['mu', 'sigma'],
    uniform: ['low', 'high'],
    truncatedNormal: ['loc', 'scale', 'low', 'high'],
  }
  const bases: Readonly<Record<string, string>> = {
    halfNormal: 'half-normal',
    halfCauchy: 'half-cauchy',
    uniform: 'uniform',
    truncatedNormal: 'truncated-normal',
  }
  const values = Object.fromEntries((names[fam] ?? []).map((p) => [p, constParam(cx, node, p, `the ${p}`)]))
  const cells: Cell[] = []
  const vars: string[] = []
  const logCells: Cell[] = []
  const site = node.dims.length === 0 ? undefined : (i: number): number => i
  for (let i = 0; i < n; i++) {
    const p = (k: string): number => (values[k] as number[])[i] as number
    const name = cx.g.addVariable({
      name: cx.elementName(node.name, node.dims, i),
      family: { kind: 'gaussian' },
    })
    const low = fam === 'uniform' || fam === 'truncatedNormal' ? p('low') : 0
    const high = fam === 'uniform' || fam === 'truncatedNormal' ? p('high') : Number.POSITIVE_INFINITY
    if (fam === 'logNormal' && !(p('sigma') > 0))
      throw new Refusal('data', `the sigma of '${node.name}' must be positive`)
    const positive = fam === 'halfNormal' || fam === 'halfCauchy' || fam === 'logNormal' || low >= 0
    // The exact density as a numeric site (`prior-site`): its message is the moment match against the current cavity at every
    // sweep, so the graph holds the prior itself. Infinite bounds are left out of the parameters (they are JSON).
    const priorFactor = (variable: string, log: boolean): FactorSpec => ({
      operator: 'prior-site',
      args: [variable],
      params: {
        y: 0,
        base: bases[fam] as string,
        log: log ? 1 : 0,
        ...(names[fam]?.includes('scale') ? { scale: p('scale') } : {}),
        ...(fam === 'truncatedNormal' ? { loc: p('loc') } : {}),
        ...(Number.isFinite(low) && (fam === 'uniform' || fam === 'truncatedNormal') ? { low } : {}),
        ...(Number.isFinite(high) && (fam === 'uniform' || fam === 'truncatedNormal') ? { high } : {}),
      },
    })
    if (positive) {
      const l = cx.g.auxName(node.name, site === undefined ? undefined : site(i))
      cx.g.addVariable({ name: l, family: { kind: 'gaussian' } })
      if (fam === 'logNormal') cx.g.addFactor(gaussianPrior(l, p('mu'), p('sigma') ** 2), l)
      else cx.g.addFactor(priorFactor(l, true), l)
      cx.g.need('apply')
      cx.g.addFactor({ operator: 'apply', args: [name, l], params: { fn: 'exp' } }, name)
      cx.logScale.set(name, l)
      logCells.push(cx.g.varAtom(l))
    } else cx.g.addFactor(priorFactor(name, false), name)
    cells.push(cx.g.varAtom(name))
    vars.push(name)
  }
  cx.syms.set(node.name, { dims: node.dims, cells })
  // A positive value carried on its log (`tau = exp(l)`): the log is addressable, so a log link reads it directly (a rate or a scale).
  if (logCells.length > 0) cx.syms.set(logName(node.name), { dims: node.dims, cells: logCells })
  cx.layout(node, 'latent', 'gaussian', node.dims.length === 0 ? 'scalar' : 'elements', vars, true)
}

/**
 * The constraint of a Gaussian latent as comparison factors on its variables: `positive`, `nonNegative`, `greaterThan`,
 * `lessThan` and `interval` with known scalar bounds truncate each element (a `greater-than` or `is-between` factor with an
 * observed true outcome, whose moment matching is the truncated Gaussian), and `ordered` says each element is above the
 * previous one (a `greater-than` on the difference of neighbours). The prior is not renormalised, as in Stan.
 */
export function truncate(cx: Context, node: ModelNode & { readonly kind: 'latent' }): void {
  const c = node.constraint
  if (c === undefined) return
  const cells = cx.syms.get(node.name)?.cells ?? []
  const flag = (x: string, operator: 'greater-than' | 'is-between', params: Record<string, Json>, i: number): void => {
    const name = cx.g.auxName(node.name, i)
    cx.g.need(operator)
    cx.g.addVariable({ name, family: { kind: 'bernoulli' }, observed: 1 })
    cx.g.addFactor({ operator, args: [name, x], params }, name)
  }
  const number = (v: Json): number => {
    if (typeof v !== 'number')
      throw new Refusal('constraint', `the bound of the constraint of '${node.name}' must be one known number`)
    return v
  }
  if (c.kind === 'ordered') {
    if (node.dims.length !== 1)
      throw new Refusal('constraint', `'${node.name}' is ordered over [${node.dims}]: one dimension is mapped`)
    for (let i = 0; i + 1 < cells.length; i++) {
      const gap = combine(affOf(cells[i + 1] as Cell), affOf(cells[i] as Cell), -1)
      flag(cx.g.materialise(gap, node.name, i) as string, 'greater-than', { threshold: 0 }, i)
    }
    return
  }
  const bounds =
    c.kind === 'positive' || c.kind === 'nonNegative'
      ? { lower: 0, upper: Number.POSITIVE_INFINITY }
      : c.kind === 'greaterThan'
        ? { lower: number(c.low), upper: Number.POSITIVE_INFINITY }
        : c.kind === 'lessThan'
          ? { lower: Number.NEGATIVE_INFINITY, upper: number(c.high) }
          : c.kind === 'interval'
            ? { lower: number(c.low), upper: number(c.high) }
            : undefined
  if (bounds === undefined)
    throw new Refusal(
      'constraint',
      `'${node.name}' has the constraint '${c.kind}': only ordered and bounds on a Gaussian are mapped`,
    )
  for (const [i, cell] of cells.entries()) {
    const x = cx.g.materialise(affOf(cell), node.name, node.dims.length === 0 ? undefined : i) as string
    if (bounds.upper === Number.POSITIVE_INFINITY) flag(x, 'greater-than', { threshold: bounds.lower }, i)
    else flag(x, 'is-between', { lower: Math.max(bounds.lower, -1e12), upper: bounds.upper }, i)
  }
}

export function latentNormal(cx: Context, node: ModelNode & { readonly kind: 'latent' }): void {
  const shape = cx.shapeOf(node.dims)
  const n = prod(shape)
  const loc = cx.over(paramSym(cx, node.dist, 'loc'), node.dims)
  const scaleExpr = node.dist.params['scale']
  if (scaleExpr !== undefined && !cx.isConst(scaleExpr)) {
    if (cx.plan.packed.has(node.name)) latentSharedPrecision(cx, node, loc)
    else if (precisionExpr(cx.idiom, scaleExpr) === undefined) latentRandomScale(cx, node, loc, scaleExpr)
    else latentRandomPrecision(cx, node, loc)
    return
  }
  const scale = constParam(cx, node, 'scale', 'the scale')
  for (const s of scale)
    if (!(s > 0)) throw new Refusal('data', `the scale of '${node.name}' is ${s}: it must be positive`)
  const group = cx.plan.groups.get(node.name)
  if (group !== undefined) {
    latentGrouped(cx, node, group)
    return
  }
  if (cx.plan.packed.has(node.name)) {
    const mean = loc.map((c) => {
      if (!isConstCell(c))
        throw new Refusal('operator', `'${node.name}' is read as a vector by a matmul, so its prior mean must be known`)
      return c.aff.offset
    })
    const cov = new Array<number>(n * n).fill(0)
    for (let i = 0; i < n; i++) cov[i * n + i] = (scale[i] as number) ** 2
    cx.g.addVariable({ name: node.name, family: { kind: 'gaussian-mv', size: n } })
    cx.g.addFactor(gaussianMvPrior(node.name, mean, cov), node.name)
    cx.syms.set(node.name, {
      dims: node.dims,
      cells: Array.from({ length: n }, (_, j) => cx.g.compAtom(node.name, j, n)),
    })
    cx.layout(node, 'latent', 'gaussian-mv', 'packed', [node.name], true)
    return
  }
  const cells: Cell[] = []
  const vars: string[] = []
  for (let i = 0; i < n; i++) {
    const name = cx.g.addVariable({
      name: cx.elementName(node.name, node.dims, i),
      family: { kind: 'gaussian' },
    })
    const l = loc[i] as Cell
    const variance = (scale[i] as number) ** 2
    if (l.kind === 'lookup') {
      // A mean chosen by a discrete variable (a mixture of components): a gate over one Gaussian prior per state, over the
      // variable and the random means of some states.
      cx.g.need('gate')
      const gate = new GateBuilder(l.variable, [name])
      for (const [k, v] of l.values.entries()) {
        const item = l.items?.[k]
        let mean: number | { readonly const: number } = { const: v }
        if (item !== undefined && item.terms.size > 0)
          mean = gate.slot(cx.g.materialise(item, node.name, node.dims.length === 0 ? undefined : i) as string)
        gate.add({ operator: 'gaussian', args: [0, mean], params: { precision: 1 / variance } })
      }
      cx.g.addFactor(gate.factor(), name)
    } else if (isConstCell(l)) cx.g.addFactor(gaussianPrior(name, l.aff.offset, variance), name)
    else {
      const m = cx.g.materialise(affOf(l), node.name, node.dims.length === 0 ? undefined : i)
      cx.g.addFactor({ operator: 'gaussian', args: [name, m as string], params: { precision: 1 / variance } }, name)
    }
    cells.push(cx.g.varAtom(name))
    vars.push(name)
  }
  cx.finishLatent(node, 'gaussian', cells, vars)
}

/**
 * A scalar normal latent that belongs to a group (`findGroups`): the first member made creates the joint Gaussian vector of
 * the group with a diagonal prior and a scalar readout variable for each member (the marginal of its coordinate, which is
 * what the posterior reads), and the members' cells are coordinates of the vector, so a predictor over several of them is
 * one dot product on it.
 */
export function latentGrouped(
  cx: Context,
  node: ModelNode & { readonly kind: 'latent' },
  group: { readonly vector: string; readonly members: readonly string[] },
): void {
  if (!cx.groupsMade.has(group.vector)) {
    cx.groupsMade.add(group.vector)
    const size = group.members.length
    const mean: number[] = []
    const cov = new Array<number>(size * size).fill(0)
    group.members.forEach((m, k) => {
      const member = cx.nodes.get(m) as ModelNode & { readonly kind: 'latent' }
      const loc = constParam(cx, member, 'loc', 'the mean')[0] as number
      const sd = constParam(cx, member, 'scale', 'the scale')[0] as number
      if (!(sd > 0)) throw new Refusal('data', `the scale of '${m}' is ${sd}: it must be positive`)
      mean.push(loc)
      cov[k * size + k] = sd * sd
    })
    cx.g.addVariable({ name: group.vector, family: { kind: 'gaussian-mv', size } })
    cx.g.auxiliary.push(group.vector)
    cx.g.addFactor(gaussianMvPrior(group.vector, mean, cov), group.vector)
    group.members.forEach((m, k) => {
      cx.g.addVariable({ name: m, family: { kind: 'gaussian' } })
      const a = new Array<number>(size).fill(0)
      a[k] = 1
      cx.g.addFactor({ operator: 'dot-constant', args: [m, group.vector], params: { a } }, m)
      cx.syms.set(m, { dims: [], cells: [cx.g.compAtom(group.vector, k, size)] })
    })
  }
  cx.layout(node, 'latent', 'gaussian', 'scalar', [node.name], true)
}

/**
 * A spike and slab latent (`normalSpikeSlab`: a mixture of a wide and a narrow zero-mean normal on every element): each
 * element is a Gaussian variable (or a coordinate of one packed Gaussian vector, when a `matmul` reads it) with a
 * `mixture-site` (`mixture-row`) factor, whose EP message is exact. A scalar site speaks first through the engine's flat-cavity
 * policy (`needsDensity`); the row on a packed vector starts with a very wide Gaussian message (a hundred slab standard
 * deviations), so the other factors have a proper cavity to begin from, and its own first update replaces it.
 */
export function latentSpikeSlab(cx: Context, node: ModelNode & { readonly kind: 'latent' }): void {
  const n = prod(cx.shapeOf(node.dims))
  if (['inclusion', 'slab', 'spike'].some((p) => !paramIsConst(cx, node, p))) {
    latentSpikeSlabSwitch(cx, node, n)
    return
  }
  const inclusion = constParam(cx, node, 'inclusion', 'the inclusion probability')
  const slab = constParam(cx, node, 'slab', 'the slab scale')
  const spike = constParam(cx, node, 'spike', 'the spike scale')
  for (let i = 0; i < n; i++)
    if (!(slab[i] !== undefined && spike[i] !== undefined && (slab[i] as number) > 0 && (spike[i] as number) > 0))
      throw new Refusal('data', `the scales of '${node.name}' must be positive`)
  const mixture = (i: number): Record<string, Json> => ({
    weights: [inclusion[i] as number, 1 - (inclusion[i] as number)],
    means: [0, 0],
    sds: [slab[i] as number, spike[i] as number],
  })
  const wide = (i: number): number => (100 * Math.max(slab[i] as number, spike[i] as number)) ** 2
  if (cx.plan.packed.has(node.name)) {
    cx.g.need('mixture-row')
    cx.g.addVariable({ name: node.name, family: { kind: 'gaussian-mv', size: n } })
    for (let i = 0; i < n; i++) {
      const a = new Array<number>(n).fill(0)
      a[i] = 1
      // The row starts with a vague message in its own coordinate, so the other factors have a proper cavity to begin from.
      const start = [...new Array<number>(n).fill(0), ...new Array<number>(n * n).fill(0)]
      start[n + i * n + i] = 1 / wide(i)
      cx.g.addFactor(
        { operator: 'mixture-row', args: [node.name], params: { a, ...mixture(i) }, start: [start] },
        cx.elementName(node.name, node.dims, i),
      )
    }
    cx.syms.set(node.name, {
      dims: node.dims,
      cells: Array.from({ length: n }, (_, j) => cx.g.compAtom(node.name, j, n)),
    })
    cx.layout(node, 'latent', 'gaussian-mv', 'packed', [node.name], true)
    return
  }
  const cells: Cell[] = []
  const vars: string[] = []
  for (let i = 0; i < n; i++) {
    const name = cx.g.addVariable({
      name: cx.elementName(node.name, node.dims, i),
      family: { kind: 'gaussian' },
    })
    cx.g.addFactor({ operator: 'mixture-site', args: [name], params: mixture(i) }, name)
    cells.push(cx.g.varAtom(name))
    vars.push(name)
  }
  cx.finishLatent(node, 'gaussian', cells, vars)
}

/**
 * A spike and slab latent with a random inclusion probability (a beta latent) or random slab or spike widths (the precision idiom
 * of a gamma latent), the switch summed out: each element is a Gaussian variable under the gate of its spike and slab Gaussians
 * over a hidden Bernoulli switch (`hiddenSwitch`), the same graph as the explicit-switch form with the switch left out of the
 * posterior. A vector that a `matmul` reads (one joint Gaussian) has one covariance, so it is refused.
 */
export function latentSpikeSlabSwitch(cx: Context, node: ModelNode & { readonly kind: 'latent' }, n: number): void {
  if (cx.plan.packed.has(node.name))
    throw new Refusal(
      'operator',
      `'${node.name}' is read as a vector by a matmul and has a random inclusion or scale: a joint Gaussian vector has one covariance, so they must be known`,
    )
  const inclusion = cx.over(paramSym(cx, node.dist, 'inclusion'), node.dims)
  // The precision of each element's slab and spike: a known one, or the gamma variable's multiple.
  const precisions = (param: 'slab' | 'spike'): ({ readonly sd: number } | Precision)[] =>
    paramIsConst(cx, node, param)
      ? constParam(cx, node, param, `the ${param} scale`).map((sd) => {
          if (!(sd > 0)) throw new Refusal('data', `the ${param} scale of '${node.name}' is ${sd}: it must be positive`)
          return { sd }
        })
      : randomPrecisions(cx, node, param)
  const slab = precisions('slab')
  const spike = precisions('spike')
  const cells: Cell[] = []
  const vars: string[] = []
  for (let i = 0; i < n; i++) {
    const index = node.dims.length === 0 ? undefined : i
    const name = cx.g.addVariable({
      name: cx.elementName(node.name, node.dims, i),
      family: { kind: 'gaussian' },
    })
    const z = hiddenSwitch(cx, node, inclusion[i] as Cell, index)
    const gate = new GateBuilder(z, [name])
    const branch = (p: { readonly sd: number } | Precision): GateBranch => {
      if ('sd' in p) return { operator: 'gaussian', args: [0, { const: 0 }], params: { precision: 1 / p.sd ** 2 } }
      const at = gate.slot(p.variable)
      cx.g.need('gaussian-gamma-precision')
      return precisionBranch([0, { const: 0 }, at], p)
    }
    // State 0 of the switch is the spike, state 1 the slab (the probability of the Bernoulli is the inclusion).
    gate.add(branch(spike[i] as never))
    gate.add(branch(slab[i] as never))
    cx.g.need('gate')
    cx.g.addFactor(gate.factor(), name)
    cells.push(cx.g.varAtom(name))
    vars.push(name)
  }
  cx.finishLatent(node, 'gaussian', cells, vars)
}

/**
 * A Gaussian latent whose scale is a random positive value (a half-normal or half-Cauchy sd of a hierarchy, `exp` of a
 * Gaussian), written in its non-centred form: `x = loc + scale * z` with a standard normal `z`, the product of two Gaussian
 * variables (the `product` factor of the engine). A scale that is not an affine Gaussian value is refused with the
 * precision idiom named.
 */
export function latentRandomScale(
  cx: Context,
  node: ModelNode & { readonly kind: 'latent' },
  loc: readonly Cell[],
  scaleExpr: ExprSpec,
): void {
  // Not an affine form of Gaussian variables (a gamma latent, a product of random values): the precision idiom is the route.
  const scales = probe(cx, () => cx.over(evalSym(cx, stripClip(cx.idiom, scaleExpr)), node.dims))
  if (scales === NOT_MINE || scales.some((c) => c.kind !== 'aff')) {
    latentRandomPrecision(cx, node, loc)
    return
  }
  cx.g.need('product')
  const cells: Cell[] = []
  const vars: string[] = []
  for (const [i, c] of scales.entries()) {
    const index = node.dims.length === 0 ? undefined : i
    const name = cx.elementName(node.name, node.dims, i)
    const z = cx.g.auxName(node.name, index)
    cx.g.addVariable({ name: z, family: { kind: 'gaussian' } })
    cx.g.addFactor(gaussianPrior(z, 0, 1), z)
    const zAtom = cx.g.atomKey({ kind: 'var', name: z })
    cx.g.atoms.set(zAtom, { kind: 'var', name: z })
    const scale = affOf(c)
    const spread = mulAff(cx, scale, { terms: new Map([[zAtom, 1]]), offset: 0 }, i)
    cx.g.materialise(combine(affOf(loc[i] as Cell), spread, 1), node.name, index, name)
    cells.push(cx.g.varAtom(name))
    vars.push(name)
  }
  cx.finishLatent(node, 'gaussian', cells, vars)
}

/**
 * A Gaussian vector (read as a vector by a matmul) whose coefficients have the precisions `s_j lambda + o_j` of one shared
 * gamma variable (a smoothing parameter): one `gaussian-mv` variable with one `gaussian-mv-precision-scale` factor, so the
 * posterior keeps the dense correlations of the coefficients and the multiplier is inferred from them.
 */
export function latentSharedPrecision(
  cx: Context,
  node: ModelNode & { readonly kind: 'latent' },
  loc: readonly Cell[],
): void {
  if (!loc.every((c) => isConstCell(c) && c.aff.offset === 0))
    throw new Refusal(
      'operator',
      `'${node.name}' has a shared random precision and a mean: the vector with a smoothing parameter must have a zero known mean`,
    )
  const taus = randomPrecisions(cx, node)
  const shared = taus[0]?.variable
  if (shared === undefined || taus.some((t) => t.variable !== shared))
    throw new Refusal(
      'operator',
      `the precisions of '${node.name}' must be multiples of one gamma variable plus constants (a vector with several random precisions is the ARD form, one variable per coefficient)`,
    )
  cx.g.need('gaussian-mv-precision-scale')
  const n = taus.length
  cx.g.addVariable({ name: node.name, family: { kind: 'gaussian-mv', size: n } })
  cx.g.addFactor(
    {
      operator: 'gaussian-mv-precision-scale',
      args: [node.name, shared],
      params: { scale: taus.map((t) => t.scale), offset: taus.map((t) => t.offset) },
    },
    node.name,
  )
  cx.syms.set(node.name, {
    dims: node.dims,
    cells: Array.from({ length: n }, (_, j) => cx.g.compAtom(node.name, j, n)),
  })
  cx.layout(node, 'latent', 'gaussian-mv', 'packed', [node.name], true)
}

/** A Gaussian latent with a gamma precision: a `gaussian-gamma-precision` factor per element (the mean-field hierarchy of Bishop 2006, 10.2). */
export function latentRandomPrecision(
  cx: Context,
  node: ModelNode & { readonly kind: 'latent' },
  loc: readonly Cell[],
): void {
  if (cx.plan.packed.has(node.name))
    throw new Refusal(
      'operator',
      `'${node.name}' is read as a vector by a matmul and has a random scale: a joint Gaussian vector has one covariance, so its scale must be known`,
    )
  const taus = randomPrecisions(cx, node)
  const cells: Cell[] = []
  const vars: string[] = []
  for (const [i, tau] of taus.entries()) {
    const name = cx.g.addVariable({
      name: cx.elementName(node.name, node.dims, i),
      family: { kind: 'gaussian' },
    })
    const l = loc[i] as Cell
    const mean = cx.g.portOf(l, node.name, node.dims.length === 0 ? undefined : i)
    cx.g.addFactor(precisionFactor(name, mean, tau), name)
    cells.push(cx.g.varAtom(name))
    vars.push(name)
  }
  cx.finishLatent(node, 'gaussian', cells, vars)
}
