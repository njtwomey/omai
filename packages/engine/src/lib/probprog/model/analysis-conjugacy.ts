/**
 * Conjugacy found in a spec, as records a person can read and a tool can act on: for each latent, whether the nodes that
 * depend on it use it in a conjugate form (the parameter itself, an `at` lookup of it by a group index, an affine map of a
 * Gaussian latent, a known multiple of a gamma rate, the variance behind a normal scale), through the plates it is summed
 * over, with latent children as well as observed ones (hierarchical conjugacy: group means with a conjugate
 * hyperprior), and the joint normal and inverse-gamma pair for a mean and variance that share a scale. Every record says
 * which sites, which pair, the closed form of the posterior, the latents the closed form is conditional on, the
 * conditions that make it exact and whether each holds, and whether the `collapse` handler can take the site out. Nothing
 * is run: the records are what the spec says, and the tests check each claimed closed form against the density.
 *
 * @index function conjugacyRecords: conjugate pairs through plates, lookups, affine maps, hierarchies and joint priors.
 */

import { CONJUGATES, type ConjugatePair } from '../../dists/index.ts'
import { isAffineIn, type LatentMemo, latentsIn, type NodesByName, nodeExprs, useOf } from './analysis-exprs.ts'
import { exprText } from './expr-text.ts'
import { canonicalDims } from './ids.ts'
import type { ExprSpec, LatentNode, ModelNode, ModelSpec, ObservedNode } from './spec.ts'

/** How a dependent node takes a latent: the mechanism that keeps the pair conjugate. */
export type ConjugacyVia =
  /** The parameter is the latent itself (broadcast over the plates the node has and the latent lacks). */
  | 'direct'
  /** The parameter is an `at` lookup of the latent by an index: the groups of a grouped model. */
  | 'index'
  /** The parameter is an affine map of a Gaussian latent with coefficients free of it (an offset, a design matrix). */
  | 'affine'
  /** The parameter is a known multiple of the latent (an exposure on a Poisson rate). */
  | 'scaled'
  /** A normal scale that is the square root of the latent (a prior on a variance). */
  | 'variance'

/** One dependent node of a latent and how it uses it. */
export interface ConjugacyLink {
  readonly node: string
  /** `observed` for a likelihood, `latent` when a latent's own prior is built on this one (a hierarchy). */
  readonly nodeKind: 'observed' | 'latent'
  readonly family: string
  /** The family's parameter the latent enters (`loc`, `rate`, `probs`, `variance`). */
  readonly parameter: string
  readonly via: ConjugacyVia
  /** The parameter's expression as text. */
  readonly expression: string
  /** The dimensions the posterior sums over: those of the node the latent does not have (a plate sum). */
  readonly sumsOver: readonly string[]
}

/** One condition of a record and whether the spec meets it. */
export interface ConjugacyCondition {
  readonly text: string
  readonly holds: boolean
}

/** What a record is about: a prior with its children, a prior on the means of latent children, or a joint prior. */
export type ConjugacyKind = 'pair' | 'hierarchical' | 'joint-prior'

/**
 * Conjugacy of one latent (or of a joint pair of latents) with what depends on it, in a form a person or a tool can read.
 *
 * @concept model-analysis
 */
export interface ConjugacyRecord {
  readonly kind: ConjugacyKind
  /** The latent sites the closed form is for: the prior, or the two latents of a joint prior. */
  readonly sites: readonly string[]
  /** The prior family of the (first) site. */
  readonly priorFamily: string
  /** The nodes whose likelihood terms it takes, with how each uses the site. */
  readonly links: readonly ConjugacyLink[]
  /** The conjugate pairs involved, `prior-likelihood parameter` (`normal-normal loc`), without repeats. */
  readonly pairs: readonly string[]
  /** The closed-form posterior of the site given the children's values (the form, in words and symbols). */
  readonly posterior: string
  /** The latents the closed form is conditional on (hyperparameters, known scales, coefficients that are latents). */
  readonly given: readonly string[]
  /** The dimensions summed over across all links (a plate sum), without repeats. */
  readonly sumsOver: readonly string[]
  /** How many child elements it takes in total, when every dimension's size is known. */
  readonly observations?: number
  readonly conditions: readonly ConjugacyCondition[]
  /** Whether every condition holds: the closed form is exact for the site given the `given` latents. */
  readonly holds: boolean
  /** Why it is, or is not, exact, in a sentence. */
  readonly reason: string
  /** Whether the `collapse` handler can integrate the site out, and why not. */
  readonly collapse: { readonly applies: boolean; readonly reason: string }
}

const POSTERIOR: Readonly<Record<string, string>> = {
  'beta/bernoulli/probs': 'Beta(a + sum y, b + n - sum y)',
  'beta/binomial/probs': 'Beta(a + sum k, b + sum (N - k))',
  'beta/geometric/probs': 'Beta(a + n, b + sum y - n)',
  'beta/negativeBinomial/probs': 'Beta(a + n r, b + sum y)',
  'gamma/poisson/rate': 'Gamma(shape + sum y, rate + n)',
  'gamma/exponential/rate': 'Gamma(shape + n, rate + sum y)',
  'gamma/gamma/rate': 'Gamma(shape + n alpha, rate + sum y)',
  'gamma/pareto/b': 'Gamma(shape + n, rate + sum log(y / scale))',
  'normal/normal/loc':
    'Normal(m_n, s_n^2) with 1/s_n^2 = 1/s_0^2 + n/sigma^2 and m_n = s_n^2 (m_0/s_0^2 + sum y/sigma^2)',
  'normal/logNormal/mu': 'Normal as for a normal mean, on log y',
  'inverseGamma/normal/variance': 'InverseGamma(a + n/2, b + sum (y - loc)^2 / 2)',
  'dirichlet/categorical/probs': 'Dirichlet(alpha + counts)',
  'dirichlet/multinomial/probs': 'Dirichlet(alpha + sum of counts)',
  'multivariateNormal/multivariateNormal/loc':
    'Normal with precision L_0 + n S^-1 and mean solving that precision against L_0 m_0 + S^-1 sum y',
  'inverseWishart/multivariateNormal/covariance': 'InverseWishart(df + n, scale + sum (y - loc)(y - loc)^T)',
  'wishart/multivariateNormal/precision': 'Wishart(df + n, (scale^-1 + sum (y - loc)(y - loc)^T)^-1)',
  'pareto/uniform/high': 'Pareto(b + n, max(scale, max y))',
}

const pairName = (p: ConjugatePair): string => `${p.prior}-${p.likelihood} ${p.parameter}`

const SCALED: ReadonlyMap<string, string> = new Map([
  ['poisson/rate', 'Gamma(shape + sum y, rate + sum w)'],
  ['exponential/rate', 'Gamma(shape + n, rate + sum w y)'],
  ['gamma/rate', 'Gamma(shape + n alpha, rate + sum w y)'],
])

/** Everything the analysis passes around: the spec's nodes, the memo of derived latents and the sizes of dimensions. */
export interface ConjugacyEnv {
  readonly spec: ModelSpec
  readonly byName: NodesByName
  readonly memo: LatentMemo
  /** The size of a named dimension, where it is known. */
  readonly sizeOf: (dim: string) => number | undefined
}

interface Use {
  readonly link: ConjugacyLink | undefined
  readonly pair: ConjugatePair | undefined
  /** Latents the closed form is conditional on because of this use (fixed parameters, coefficients, weights). */
  readonly given: ReadonlySet<string>
  /** Why this use is not a conjugate form, when it is not. */
  readonly problem: string | undefined
  /** `form`: the use is not a conjugate form; `known`: a parameter the pair needs known depends on the latent. */
  readonly category?: 'form' | 'known'
}

function dimsOfNode(env: ConjugacyEnv, n: ModelNode): string[] {
  return canonicalDims(env.spec, n.dims)
}

/** The weights of a product `w * x` or `x * w` when the other factor is not the latent, else undefined. */
function scaledBy(e: ExprSpec, x: string): ExprSpec | undefined {
  if (e.kind !== 'op' || e.op !== 'mul') return undefined
  const [a, b] = e.args
  if (a === undefined || b === undefined) return undefined
  if (a.kind === 'ref' && a.name === x) return b
  if (b.kind === 'ref' && b.name === x) return a
  return undefined
}

function classify(
  env: ConjugacyEnv,
  prior: LatentNode,
  node: LatentNode | ObservedNode,
  param: string,
  expr: ExprSpec,
): Use {
  const x = prior.name
  const none = new Set<string>()
  const fail = (problem: string, category: 'form' | 'known' = 'form'): Use => ({
    link: undefined,
    pair: undefined,
    given: none,
    problem,
    category,
  })
  const family = node.dist.family
  const find = (parameter: string): ConjugatePair | undefined =>
    CONJUGATES.find((c) => c.likelihood === family && c.parameter === parameter && c.prior === prior.dist.family)
  const others = (e: ExprSpec): Set<string> => {
    const s = latentsIn(e, env.byName, env.memo)
    s.delete(x)
    return s
  }
  const mk = (via: ConjugacyVia, parameter: string, pair: ConjugatePair, given: Set<string>): Use => {
    const own = dimsOfNode(env, prior)
    const kidDims = dimsOfNode(env, node)
    // Fixed parameters of the pair must not depend on the latent; the rest of their latents are conditioned on.
    let known: string | undefined
    for (const f of pair.fixed) {
      const fe = node.dist.params[f]
      if (fe === undefined) continue
      if (latentsIn(fe, env.byName, env.memo).has(x)) known = `the known ${f} of '${node.name}' depends on '${x}'`
      for (const l of latentsIn(fe, env.byName, env.memo)) given.add(l)
    }
    given.delete(x)
    return {
      link: {
        node: node.name,
        nodeKind: node.kind === 'latent' ? 'latent' : 'observed',
        family,
        parameter,
        via,
        expression: exprText(expr),
        sumsOver: via === 'index' ? kidDims : kidDims.filter((d) => !own.includes(d)),
      },
      pair,
      given,
      problem: known,
      ...(known === undefined ? {} : { category: 'known' as const }),
    }
  }
  const direct = useOf(expr, x)
  if (direct !== undefined) {
    const pair = find(param)
    if (pair === undefined) return fail(`${family} has no conjugate prior of family ${prior.dist.family} on '${param}'`)
    const given = new Set<string>()
    // A lookup by a latent index (the component of a mixture) is conjugate given the index: the closed form is conditional.
    if (expr.kind === 'op' && expr.op === 'at') {
      const idx = expr.args[1]
      if (idx !== undefined) for (const l of latentsIn(idx, env.byName, env.memo)) given.add(l)
    }
    return mk(direct, param, pair, given)
  }
  // A normal scale that is the square root of the variance latent.
  if (family === 'normal' && param === 'scale' && expr.kind === 'op' && expr.op === 'sqrt') {
    const inner = expr.args[0]
    const v = inner === undefined ? undefined : useOf(inner, x)
    const pair = find('variance')
    if (v !== undefined && pair !== undefined) return mk('variance', 'variance', pair, new Set())
  }
  // A known multiple of a gamma rate: an exposure on a Poisson rate.
  const w = scaledBy(expr, x)
  if (w !== undefined && prior.dist.family === 'gamma' && SCALED.has(`${family}/${param}`)) {
    const pair = find(param)
    if (pair !== undefined && !latentsIn(w, env.byName, env.memo).has(x)) return mk('scaled', param, pair, others(w))
  }
  // An affine map of a Gaussian latent in a normal mean.
  if (family === 'normal' && param === 'loc' && prior.dist.family === 'normal') {
    const pair = find('loc')
    const vars = new Set([x])
    if (pair !== undefined && isAffineIn(expr, env.byName, env.memo, vars))
      return mk('affine', 'loc', pair, others(expr))
    if (pair !== undefined) return fail(`the mean of '${node.name}' is not affine in '${x}' (${exprText(expr)})`)
  }
  return fail(`'${node.name}' uses '${x}' in its ${param} as ${exprText(expr)}, which is not a conjugate form`)
}

/** The consumers of a latent: the latent and observed nodes (and factors) whose expressions depend on it. */
function consumers(env: ConjugacyEnv, x: string): ModelNode[] {
  return env.spec.nodes.filter(
    (n) =>
      n.name !== x &&
      (n.kind === 'latent' || n.kind === 'observed' || n.kind === 'factor') &&
      nodeExprs(n).some((e) => latentsIn(e, env.byName, env.memo).has(x)),
  )
}

function observationCount(env: ConjugacyEnv, links: readonly ConjugacyLink[]): number | undefined {
  let total = 0
  for (const l of links) {
    const node = env.spec.nodes.find((n) => n.name === l.node && n.kind === l.nodeKind)
    if (node === undefined) return undefined
    let k = 1
    for (const d of dimsOfNode(env, node)) {
      const size = env.sizeOf(d)
      if (size === undefined) return undefined
      k *= size
    }
    total += k
  }
  return total
}

function pairRecord(env: ConjugacyEnv, prior: LatentNode): ConjugacyRecord | undefined {
  const x = prior.name
  if (!CONJUGATES.some((c) => c.prior === prior.dist.family)) return undefined
  const links: ConjugacyLink[] = []
  const pairs: ConjugatePair[] = []
  const given = new Set<string>()
  const problems: string[] = []
  const categories = new Set<string>()
  const problem = (text: string, category: 'form' | 'known' | 'other'): void => {
    problems.push(text)
    categories.add(category)
  }
  for (const n of consumers(env, x)) {
    if (n.kind === 'factor') {
      problem(`the factor '${n.name}' adds an arbitrary density term in '${x}'`, 'other')
      continue
    }
    if (n.kind !== 'latent' && n.kind !== 'observed') continue
    if (n.kind === 'observed' && latentsIn(n.value, env.byName, env.memo).has(x)) {
      problem(`the value of '${n.name}' depends on '${x}'`, 'other')
      continue
    }
    let used = false
    for (const [param, expr] of Object.entries(n.dist.params)) {
      if (!latentsIn(expr, env.byName, env.memo).has(x)) continue
      used = true
      const u = classify(env, prior, n, param, expr)
      if (u.problem !== undefined) problem(u.problem, u.category ?? 'form')
      if (u.link !== undefined && u.pair !== undefined) {
        links.push(u.link)
        if (!pairs.includes(u.pair)) pairs.push(u.pair)
        for (const g of u.given) given.add(g)
      }
    }
    if (!used) problem(`'${n.name}' depends on '${x}' only through its value`, 'other')
  }
  if (links.length === 0 && problems.length === 0) return undefined
  if (links.length === 0) return undefined
  for (const e of Object.values(prior.dist.params)) for (const l of latentsIn(e, env.byName, env.memo)) given.add(l)
  given.delete(x)
  const hierarchical = links.some((l) => l.nodeKind === 'latent')
  const vias = new Set(links.map((l) => l.via))
  const sumsOver = [...new Set(links.flatMap((l) => l.sumsOver))]
  const posterior = links.some((l) => l.via === 'scaled')
    ? (SCALED.get(`${links[0]?.family}/${links[0]?.parameter}`) ?? 'Gamma with updated shape and rate')
    : pairs
        .map((p) => POSTERIOR[`${p.prior}/${p.likelihood}/${p.parameter}`] ?? `${p.prior} with updated parameters`)
        .join('; ')
  const conditions: ConjugacyCondition[] = [
    { text: 'every dependent node uses the site in a conjugate form', holds: !categories.has('form') },
    { text: 'the known parameters of the dependents do not depend on the site', holds: !categories.has('known') },
    { text: 'nothing else (a factor term, a value) depends on the site', holds: !categories.has('other') },
  ]
  const holds = problems.length === 0
  const givenList = [...given]
  const summary = `${pairs.map(pairName).join(', ')} through ${[...vias].join(', ')}`
  const reason = holds
    ? `${x} is conjugate (${summary}): its posterior is closed form${givenList.length > 0 ? ` given ${givenList.join(', ')}` : ', with nothing else to condition on'}${sumsOver.length > 0 ? `, summing over ${sumsOver.join(', ')}` : ''}`
    : `${x} would be conjugate (${summary}) but ${problems.join('; ')}`
  // A recorded derived node that reads the site is a `deterministic` site that depends on it: the handler refuses it.
  const recorded = env.spec.nodes
    .filter((n) => n.kind === 'derived' && n.record && latentsIn(n.expr, env.byName, env.memo).has(x))
    .map((n) => n.name)
  // The handler lines the site up with the last axes of its children, in the order the nodes declare their dimensions.
  const own = prior.dims
  let layout: string | undefined
  for (const l of links) {
    const kid = env.spec.nodes.find((n) => n.name === l.node && n.kind === l.nodeKind)
    const kd = kid?.dims ?? []
    if (own.length > 0 && l.via !== 'index' && !own.every((dim, i) => kd[kd.length - own.length + i] === dim))
      layout = `its dimensions (${own.join(', ')}) are not the trailing dimensions of ${l.node} (${kd.join(', ')}): the handler lines the site up with the last axes, so ${l.node} would have to be declared with dims [..., ${own.join(', ')}] (observe takes \`dims\`)`
  }
  const collapse = collapseVerdict(holds, hierarchical, pairs, [...vias], recorded, layout)
  return {
    kind: hierarchical ? 'hierarchical' : 'pair',
    sites: [x],
    priorFamily: prior.dist.family,
    links,
    pairs: pairs.map(pairName),
    posterior,
    given: givenList,
    sumsOver,
    ...(() => {
      const n = observationCount(env, links)
      return n === undefined ? {} : { observations: n }
    })(),
    conditions,
    holds,
    reason,
    collapse,
  }
}

function collapseVerdict(
  holds: boolean,
  hierarchical: boolean,
  pairs: readonly ConjugatePair[],
  vias: readonly ConjugacyVia[],
  recorded: readonly string[],
  layout: string | undefined,
): { readonly applies: boolean; readonly reason: string } {
  if (!holds) return { applies: false, reason: 'the site is not conjugate with everything that depends on it' }
  if (hierarchical)
    return {
      applies: false,
      reason: 'some dependents are latent: collapse integrates a site into observed children, so collapse those first',
    }
  if (pairs.length !== 1) return { applies: false, reason: 'the children are not all of one conjugate pair' }
  const unlike = vias.filter((v) => v !== 'direct' && v !== 'variance')
  if (unlike.length > 0)
    return {
      applies: false,
      reason: `the handler takes the parameter itself (or the variance behind a scale), not ${[...new Set(unlike)].join(' or ')}`,
    }
  if (layout !== undefined) return { applies: false, reason: layout }
  if (recorded.length > 0)
    return {
      applies: false,
      reason: `the recorded derived node ${recorded.join(', ')} reads the site, and the handler takes only sites whose dependents are observed`,
    }
  return { applies: true, reason: 'the children take the site as their parameter and are all of one conjugate pair' }
}

/** A scale expression as `c * sqrt(v)` or `sqrt(c * v)` for a known `c`: the factor's text, `1` when there is none. */
function sqrtOf(env: ConjugacyEnv, e: ExprSpec, v: string): string | undefined {
  const free = (x: ExprSpec): boolean => latentsIn(x, env.byName, env.memo).size === 0
  const isV = (x: ExprSpec): boolean => x.kind === 'ref' && x.name === v
  const isSqrtV = (x: ExprSpec): boolean =>
    x.kind === 'op' && x.op === 'sqrt' && x.args[0] !== undefined && isV(x.args[0])
  if (isSqrtV(e)) return '1'
  if (e.kind !== 'op') return undefined
  const [a, b] = e.args
  if (a === undefined) return undefined
  if (e.op === 'sqrt') {
    const inner = a
    if (inner.kind === 'op' && inner.args.length === 2) {
      const [p, q] = inner.args
      if (p === undefined || q === undefined) return undefined
      if (inner.op === 'mul' && ((isV(p) && free(q)) || (isV(q) && free(p)))) return exprText(isV(p) ? q : p)
      if (inner.op === 'div' && isV(p) && free(q)) return `1 / ${exprText(q)}`
    }
    return undefined
  }
  if (b === undefined) return undefined
  if (e.op === 'mul' && ((isSqrtV(a) && free(b)) || (isSqrtV(b) && free(a)))) return `${exprText(isSqrtV(a) ? b : a)}^2`
  if (e.op === 'div' && isSqrtV(a) && free(b)) return `1 / ${exprText(b)}^2`
  return undefined
}

function jointPriorRecords(env: ConjugacyEnv): ConjugacyRecord[] {
  const out: ConjugacyRecord[] = []
  const latents = env.spec.nodes.filter((n): n is LatentNode => n.kind === 'latent')
  for (const mu of latents) {
    if (mu.dist.family !== 'normal') continue
    const scale = mu.dist.params['scale']
    if (scale === undefined) continue
    // The prior scale of the mean must be built on one inverse-gamma latent, as the square root of a multiple of it.
    const vs = [...latentsIn(scale, env.byName, env.memo)]
    const v = vs.length === 1 ? latents.find((l) => l.name === vs[0]) : undefined
    if (v === undefined || v.dist.family !== 'inverseGamma') continue
    const factor = sqrtOf(env, scale, v.name)
    if (factor === undefined) continue
    const m0 = mu.dist.params['loc']
    const problems: string[] = []
    if (m0 !== undefined && latentsIn(m0, env.byName, env.memo).size > 0)
      problems.push(`the prior mean of '${mu.name}' depends on a latent`)
    for (const e of Object.values(v.dist.params))
      if (latentsIn(e, env.byName, env.memo).size > 0) problems.push(`the prior of '${v.name}' depends on a latent`)
    const muDims = dimsOfNode(env, mu)
    const vDims = dimsOfNode(env, v)
    if (muDims.length !== vDims.length || muDims.some((d, i) => d !== vDims[i]))
      problems.push(`'${mu.name}' and '${v.name}' are over different dimensions`)
    const links: ConjugacyLink[] = []
    for (const n of consumers(env, mu.name)) {
      if (n.kind !== 'observed') {
        problems.push(`'${n.name}' (${n.kind}) depends on '${mu.name}'`)
        continue
      }
      const loc = n.dist.params['loc']
      const sc = n.dist.params['scale']
      const usesMu = loc !== undefined && useOf(loc, mu.name) === 'direct'
      const sameScale = sc !== undefined && sqrtOf(env, sc, v.name) === '1'
      if (n.dist.family !== 'normal' || !usesMu) {
        problems.push(`'${n.name}' is not Normal(${mu.name}, sqrt(${v.name}))`)
        continue
      }
      if (!sameScale)
        problems.push(
          `'${n.name}' is not Normal(${mu.name}, sqrt(${v.name})): its scale is another function of ${v.name}`,
        )
      if (latentsIn(n.value, env.byName, env.memo).size > 0)
        problems.push(`the value of '${n.name}' depends on a latent`)
      const kid = dimsOfNode(env, n)
      links.push({
        node: n.name,
        nodeKind: 'observed',
        family: 'normal',
        parameter: 'loc',
        via: 'direct',
        expression: exprText(loc),
        sumsOver: kid.filter((d) => !muDims.includes(d)),
      })
    }
    for (const n of consumers(env, v.name))
      if (n.name !== mu.name && !links.some((l) => l.node === n.name))
        problems.push(`'${n.name}' also depends on '${v.name}'`)
    if (links.length === 0) continue
    const holds = problems.length === 0
    const observations = observationCount(env, links)
    out.push({
      kind: 'joint-prior',
      sites: [mu.name, v.name],
      priorFamily: 'normalInverseGamma',
      links,
      pairs: ['normal-inverse-gamma on (mean, variance)'],
      posterior: `NormalInverseGamma: ${v.name} | y ~ InverseGamma(a + n/2, b + (sum (y - ybar)^2 + k_0 n (ybar - m_0)^2 / (k_0 + n)) / 2), ${mu.name} | ${v.name}, y ~ Normal((k_0 m_0 + n ybar) / (k_0 + n), ${v.name} / (k_0 + n)), with the prior variance of ${mu.name} equal to ${v.name} / k_0 (a factor ${factor} on the scale)`,
      given: [],
      sumsOver: [...new Set(links.flatMap((l) => l.sumsOver))],
      ...(observations === undefined ? {} : { observations }),
      conditions: [
        { text: `${mu.name} is Normal with a scale proportional to sqrt(${v.name}) (factor ${factor})`, holds: true },
        {
          text: `${v.name} is InverseGamma with known parameters, ${mu.name} has a known prior mean`,
          holds: !problems.some((p) => p.includes('depends on a latent')),
        },
        {
          text: `every observation is Normal(${mu.name}, sqrt(${v.name}))`,
          holds: !problems.some((p) => p.includes('is not Normal')),
        },
        {
          text: `nothing else depends on ${mu.name} or ${v.name}`,
          holds: !problems.some((p) => p.includes('also depends') || p.includes("depends on '")),
        },
      ],
      holds,
      reason: holds
        ? `${mu.name} and ${v.name} are jointly conjugate (normal-inverse-gamma): their joint posterior is closed form, and integrating both out leaves a Student-t likelihood of the observations`
        : `${mu.name} and ${v.name} have the normal-inverse-gamma form but ${problems.join('; ')}`,
      collapse: {
        applies: false,
        reason:
          'the handler collapses one site at a time into its children; a joint prior is two sites that depend on each other',
      },
    })
  }
  return out
}

/**
 * The conjugacy of a spec's latents with what depends on them, as {@link ConjugacyRecord}s, one per conjugate latent (a
 * pair or a hierarchy) and one per joint normal and inverse-gamma prior.
 *
 * @param env - The spec, its value nodes by name, the memo of derived nodes' latents and the sizes of dimensions.
 * @returns The records, in the order of the latents.
 */
export function conjugacyRecords(env: ConjugacyEnv): ConjugacyRecord[] {
  const out: ConjugacyRecord[] = []
  for (const n of env.spec.nodes) {
    if (n.kind !== 'latent') continue
    const r = pairRecord(env, n)
    if (r !== undefined) out.push(r)
  }
  out.push(...jointPriorRecords(env))
  return out
}
