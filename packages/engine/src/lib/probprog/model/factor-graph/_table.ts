/**
 * The ordered mapping tables of the compiler: which mapping takes a node, in what order. `latentMappings` is asked for every latent
 * node, `observedNodeMappings` and `observedRowMappings` for every observed node, `observedElementMappings` for each element of an
 * observed node over its dimensions. The first entry of a table whose pattern the node is maps it, and the order is part of the
 * meaning: an entry says why it sits where it does when an earlier or later neighbour would also take the node. A mapping that
 * recognises by trying answers `NOT_MINE` and the graph is restored (`_mapping.ts`), so a failed recognition leaves nothing behind.
 * Every entry carries a line saying what it maps and the factor operators it can emit; the list of what the compiler maps in
 * `docs/probprog/` is generated from them.
 */

import type { FamilyKind } from '../../../message-passing/index.ts'
import type { ExprSpec } from '../spec.ts'
import { type Cell, prod } from './_cells.ts'
import type { Context } from './_context.ts'
import { expArgument, precisionExpr, unwrap } from './_idioms.ts'
import { type LatentNode, type Mapping, type MappingTable, mapFirst, NOT_MINE, type ObservedNode } from './_mapping.ts'
import { latentDirichlet, latentDiscrete } from './_mappings/latent-discrete.ts'
import { latentNormal, latentProjected, latentSpikeSlab, truncate } from './_mappings/latent-gaussian.ts'
import {
  latentGammaRate,
  latentKnownPrior,
  latentNormalGamma,
  latentRandomConcentration,
} from './_mappings/latent-positive.ts'
import {
  latentCorrelation,
  latentMultivariateNormal,
  latentMvnWishart,
  latentWishart,
  observedBivariateNormal,
  observedMvnWishart,
  wishartOf,
} from './_mappings/matrix.ts'
import {
  observedBernoulli,
  observedConjugate,
  observedFailures,
  observedGammaRate,
  observedMixtureWeight,
} from './_mappings/observed-conjugate.ts'
import {
  observedBernoulliTable,
  observedSwitch,
  observedSwitchLogScale,
  switchedCells,
  valueMovedToMean,
} from './_mappings/observed-gates.ts'
import {
  observedGaussian,
  observedHeavy,
  observedLogScale,
  observedNormalGamma,
  observedRandomPrecision,
  observedRandomScale,
} from './_mappings/observed-gaussian.ts'
import { observedHurdle, observedMulticlass, observedNumeric, observedPair } from './_mappings/observed-numeric.ts'
import { plate } from './_mappings/plates.ts'
import { discreteParents, evalNamedValue } from './_params.ts'
import { Refusal } from './_refusal.ts'

/** A factor-graph name for the family of a distribution family, for the families the compiler maps. */
export const MESSAGE_FAMILY: Readonly<Record<string, FamilyKind>> = {
  normal: 'gaussian',
  normalSpikeSlab: 'gaussian',
  gamma: 'gamma',
  inverseGamma: 'gamma',
  exponential: 'gamma',
  beta: 'beta',
  dirichlet: 'dirichlet',
  bernoulli: 'bernoulli',
  categorical: 'discrete',
  multivariateNormal: 'gaussian-mv',
}

/**
 * The families of a prior with no message family that are projected onto a Gaussian once (`latentProjected`): a positive
 * scale (half-normal, half-Cauchy, log-normal, uniform or truncated normal on the positive half line) as a Gaussian on its
 * log, any other bounded or truncated value as a Gaussian on the value.
 */
export const PROJECTED_PRIORS: ReadonlySet<string> = new Set([
  'halfNormal',
  'halfCauchy',
  'logNormal',
  'uniform',
  'truncatedNormal',
])

// ----- latent nodes -----

/** Whether a latent's constraint is a truncation of a normal (the one constraint beyond the family's support that is mapped). */
const truncatedNormal = (node: LatentNode): boolean =>
  node.constraint !== undefined && node.constraint.kind !== 'real' && node.dist.family === 'normal'

/** The families whose latents are an inverse-gamma or gamma variable. */
const isGamma = (node: LatentNode): boolean => node.dist.family === 'gamma' || node.dist.family === 'inverseGamma'

/**
 * The mappings of latent nodes, in priority order, one per line of what a latent can be. The first entry that takes the node maps it.
 * The refusals sit where the first node they refuse could otherwise be taken by a later entry.
 */
export const latentMappings: MappingTable<[LatentNode], void> = [
  {
    name: 'constraint-refused',
    maps: 'a constraint beyond the family support (a simplex, a positive-definite matrix) is refused, but for a truncated normal',
    order: 'first: a constrained node is never mapped by the family entries below',
    requires: [],
    matches: (_cx, node) => node.constraint !== undefined && node.constraint.kind !== 'real' && !truncatedNormal(node),
    map: (_cx, node) => {
      throw new Refusal(
        'constraint',
        `'${node.name}' has the constraint '${(node.constraint as { readonly kind: string }).kind}': a message family lives on the family's own support, so a constraint beyond it is not expressible`,
      )
    },
  },
  {
    name: 'projected-prior',
    maps: 'a half-normal, half-Cauchy, log-normal, uniform or truncated normal prior, projected onto a Gaussian (on the log of a positive scale)',
    order: 'before the family lookup: these families have no message family of their own',
    families: [...PROJECTED_PRIORS],
    requires: ['prior'],
    matches: (_cx, node) => PROJECTED_PRIORS.has(node.dist.family),
    map: (cx, node) => latentProjected(cx, node),
  },
  {
    name: 'wishart',
    maps: 'a Wishart or inverse-Wishart matrix: one variable of the Wishart family, read only as the precision or covariance of a multivariate normal',
    families: ['wishart', 'inverseWishart'],
    requires: ['prior'],
    matches: (_cx, node) => node.dist.family === 'wishart' || node.dist.family === 'inverseWishart',
    map: (cx, node) => latentWishart(cx, node),
  },
  {
    name: 'correlation',
    maps: 'a correlation matrix of order two, as the beta variable of (rho + 1) / 2',
    families: ['correlation'],
    requires: ['prior'],
    matches: (_cx, node) => node.dist.family === 'correlation',
    map: (cx, node) => latentCorrelation(cx, node),
  },
  {
    name: 'bivariate-normal-refused',
    maps: 'a latent bivariate normal is refused: only observed vectors are mapped',
    families: ['bivariateNormal'],
    requires: [],
    matches: (_cx, node) => node.dist.family === 'bivariateNormal',
    map: (_cx, node) => {
      throw new Refusal(
        'structure',
        `'${node.name}' is a latent bivariateNormal: only observed vectors are mapped (write the vectors with a Wishart precision, or observe them)`,
      )
    },
  },
  {
    name: 'mvn-wishart',
    maps: 'a multivariate normal whose precision (or covariance) is a Wishart latent: the normal-Wishart pair with an mvn-normal-wishart factor',
    order: 'before the plain multivariate normal below',
    families: ['multivariateNormal'],
    requires: ['mvn-normal-wishart', 'prior'],
    matches: (cx, node) => node.dist.family === 'multivariateNormal' && wishartOf(cx, node) !== undefined,
    map: (cx, node) => latentMvnWishart(cx, node),
  },
  {
    name: 'normal-gamma-mean',
    maps: 'the Gaussian mean of an activated normal-gamma pair: it has no variable of its own, the pair is the gamma precision variable',
    order: 'before the gamma entry: the pair is activated by the precision, which the walk reaches first',
    requires: [],
    matches: (cx, node) => cx.normalGammaByMean.has(node.name),
    map: (cx, node) => {
      const pair = cx.normalGammaByMean.get(node.name)
      cx.layout(node, 'latent', 'normal-gamma', 'scalar', [(pair as { readonly variable: string }).variable], true, {
        part: 'mean',
      })
    },
  },
  {
    name: 'normal-gamma-precision',
    maps: 'a gamma or inverse-gamma precision (or variance) that a Gaussian mean and its observations share: one normal-gamma variable with a gaussian-normal-gamma factor per observation',
    order:
      'before the family lookup (the pair replaces the gamma and the mean); NOT_MINE when the plan found no pair, then the gamma is an ordinary one',
    families: ['gamma', 'inverseGamma'],
    requires: ['gaussian-normal-gamma', 'prior'],
    matches: (_cx, node) => isGamma(node),
    map: (cx, node) => {
      const mapped = latentNormalGamma(cx, node)
      return mapped === NOT_MINE ? NOT_MINE : undefined
    },
  },
  {
    name: 'family-refused',
    maps: 'a family with no message family and no projection is refused, naming the engine families',
    order: 'after the entries of the families that have no message family of their own, before the message families',
    requires: [],
    matches: (_cx, node) => MESSAGE_FAMILY[node.dist.family] === undefined,
    map: (_cx, node) => {
      const fam = node.dist.family
      throw new Refusal(
        'family',
        `'${node.name}' is ${fam}: the engine has message families gaussian, gamma, beta, bernoulli, discrete and dirichlet, and none for ${fam}`,
      )
    },
  },
  {
    name: 'normal',
    maps: 'a normal: a Gaussian variable per element, with a known or random mean and scale (a precision, a random scale, a switch), grouped or packed where the plan says; a truncation constrains it',
    families: ['normal'],
    requires: [
      'apply',
      'dot-constant',
      'gate',
      'gaussian',
      'gaussian-gamma-precision',
      'gaussian-mv-precision-scale',
      'greater-than',
      'product',
      'prior',
    ],
    matches: (_cx, node) => node.dist.family === 'normal',
    map: (cx, node) => {
      latentNormal(cx, node)
      if (truncatedNormal(node)) truncate(cx, node)
    },
  },
  {
    name: 'multivariate-normal',
    maps: 'a multivariate normal with a known or a random mean vector: one Gaussian vector variable',
    families: ['multivariateNormal'],
    requires: ['gaussian-transition', 'mvn-wishart', 'prior'],
    matches: (_cx, node) => node.dist.family === 'multivariateNormal',
    map: (cx, node) => latentMultivariateNormal(cx, node),
  },
  {
    name: 'spike-and-slab',
    maps: 'a spike-and-slab normal: a Gaussian that a Bernoulli inclusion switches between a narrow and a wide component',
    families: ['normalSpikeSlab'],
    requires: ['gate', 'gaussian-gamma-precision', 'mixture-row', 'mixture-site', 'prior-site', 'prior'],
    matches: (_cx, node) => node.dist.family === 'normalSpikeSlab',
    map: (cx, node) => latentSpikeSlab(cx, node),
  },
  {
    name: 'discrete',
    maps: 'a Bernoulli or categorical: a discrete variable with a prior, a table over its discrete parents, a gate for a switch, or Dirichlet rows',
    families: ['bernoulli', 'categorical'],
    requires: ['bernoulli', 'discrete', 'gate', 'prior', 'stick-breaking', 'table'],
    matches: (_cx, node) => node.dist.family === 'bernoulli' || node.dist.family === 'categorical',
    map: (cx, node) => latentDiscrete(cx, node),
  },
  {
    name: 'dirichlet-random-concentration',
    maps: 'a Dirichlet whose concentration is random (a symmetric strength with a gamma or log-normal prior): a dirichlet-concentration factor',
    order: 'before the Dirichlet with a known concentration',
    families: ['dirichlet'],
    requires: ['dirichlet-concentration', 'prior'],
    matches: (cx, node) => {
      const concentration = node.dist.params['concentration']
      return node.dist.family === 'dirichlet' && concentration !== undefined && !cx.isConst(concentration)
    },
    map: (cx, node) => latentRandomConcentration(cx, node),
  },
  {
    name: 'dirichlet',
    maps: 'a Dirichlet with a known concentration: one Dirichlet variable per element with its prior',
    families: ['dirichlet'],
    requires: ['prior'],
    matches: (_cx, node) => node.dist.family === 'dirichlet',
    map: (cx, node) => latentDirichlet(cx, node),
  },
  {
    name: 'beta-random-shapes',
    maps: 'a beta with a random shape (a log-normal or gamma strength): a beta-two-shapes factor',
    order: 'before the beta with known shapes in the closed-form entry at the end',
    families: ['beta'],
    requires: ['beta-two-shapes', 'prior'],
    matches: (cx, node) =>
      node.dist.family === 'beta' && ['a', 'b'].some((k) => !cx.isConst(node.dist.params[k] as ExprSpec)),
    map: (cx, node) => latentRandomConcentration(cx, node),
  },
  {
    name: 'gamma-random-rate',
    maps: 'a gamma or inverse gamma whose rate (or scale) is a known multiple of another gamma latent: a gamma-rate factor',
    order: 'before the gamma with known parameters in the closed-form entry at the end',
    families: ['gamma', 'inverseGamma'],
    requires: ['gamma-rate', 'prior'],
    matches: (cx, node) => {
      const rate = node.dist.params[node.dist.family === 'inverseGamma' ? 'scale' : 'rate']
      return isGamma(node) && rate !== undefined && !cx.isConst(rate)
    },
    map: (cx, node) => latentGammaRate(cx, node),
  },
  {
    name: 'known-prior',
    maps: 'a beta, gamma, inverse-gamma or exponential with known parameters: one variable per element with its prior',
    order: 'last: what is left of the message families once every random-parameter form has been tried',
    families: ['beta', 'gamma', 'inverseGamma', 'exponential'],
    requires: ['prior'],
    map: (cx, node) => latentKnownPrior(cx, node, MESSAGE_FAMILY[node.dist.family] as FamilyKind),
  },
]

/** Map a latent node with the first mapping of `latentMappings` that takes it. */
export function mapLatent(cx: Context, node: LatentNode): void {
  mapFirst(latentMappings, cx, 'latent', node)
}

// ----- observed nodes -----

/** What the table of an observed node's rows works from: the values of the node over its dimensions and the discrete latents its parameters read. */
export interface Rows {
  readonly values: readonly number[]
  readonly parents: readonly string[]
}

/** One element of an observed node: its family, the variable name it would have, its value, its index in the node and the index its factors name. */
export interface Element {
  readonly fam: string
  readonly name: string
  readonly y: number
  readonly i: number
  readonly index: number | undefined
}

/**
 * The mappings of an observed node as a whole, in priority order. The last entry (`rows`) takes every node that is not one of the
 * whole-node forms before it: it maps the node row by row.
 */
export const observedNodeMappings: MappingTable<[ObservedNode], void> = [
  {
    name: 'mvn-wishart-observed',
    maps: 'an observed multivariate normal whose precision (or covariance) is a Wishart latent: an mvn-wishart factor on the observed vector',
    families: ['multivariateNormal'],
    requires: ['mvn-wishart'],
    matches: (cx, node) => node.dist.family === 'multivariateNormal' && wishartOf(cx, node) !== undefined,
    map: (cx, node) => observedMvnWishart(cx, node),
  },
  {
    name: 'bivariate-normal-observed',
    maps: 'an observed bivariate normal with a Gaussian mean and a correlation latent: an mvn-sd-corr factor',
    families: ['bivariateNormal'],
    requires: ['mvn-sd-corr'],
    matches: (_cx, node) => node.dist.family === 'bivariateNormal',
    map: (cx, node) => observedBivariateNormal(cx, node),
  },
  {
    name: 'value-moved-to-mean',
    maps: 'a Gaussian observation of a value that is a known function of a discrete variable: the value moves to the mean, a gate over the shifted forms',
    order: 'before every other observation: the node is rewritten and mapped again from the top of this table',
    families: ['normal'],
    requires: ['gate', 'gaussian-scale', 'gaussian'],
    map: (cx, node) => {
      const moved = valueMovedToMean(cx, node)
      if (moved === undefined) return NOT_MINE
      mapObserved(cx, moved)
      return undefined
    },
  },
  {
    name: 'rows',
    maps: 'every other observed node, row by row: the values are read, then the table of rows and of elements maps them',
    requires: [],
    map: (cx, node) => observedRows(cx, node),
  },
]

/** Map an observed node with the first mapping of `observedNodeMappings` that takes it. */
export function mapObserved(cx: Context, node: ObservedNode): void {
  mapFirst(observedNodeMappings, cx, 'observed node', node)
}

/**
 * The mappings of the rows of an observed node (its values read), in priority order. The last entry (`elements`) takes the node
 * element by element, through `observedElementMappings`.
 */
export const observedRowMappings: MappingTable<[ObservedNode, Rows], void> = [
  {
    name: 'normal-gamma-observed',
    maps: 'an observed Gaussian of an activated normal-gamma pair: a gaussian-normal-gamma factor on the pair variable',
    order: 'before the plate: the pair variable is not a weight vector',
    families: ['normal'],
    requires: ['gaussian-normal-gamma'],
    matches: (cx, node) => cx.normalGammaObserved.has(node.name),
    map: (cx, node, rows) => {
      const pair = cx.normalGammaObserved.get(node.name)
      if (pair === undefined) return NOT_MINE
      observedNormalGamma(cx, node, pair, rows.values)
      return undefined
    },
  },
  {
    name: 'plate',
    maps: 'rows that are i.i.d. draws of one regression on a weight vector (a Gaussian with one known scale, a probit or logistic Bernoulli): ONE plated factor holding the rows',
    order:
      'before the discrete-parent table and the elements: the plated factor replaces the per-row factors (NOT_MINE: the rows are expanded)',
    families: ['normal', 'bernoulli'],
    requires: ['linear-regression-plate', 'logistic-plate', 'probit-plate'],
    matches: (cx, node, rows) => cx.vectorise && node.dims.length > 0 && rows.parents.length === 0,
    map: (cx, node, rows) => {
      const mapped = plate(cx, node, node.dist.family, rows.values)
      return mapped === NOT_MINE ? NOT_MINE : undefined
    },
  },
  {
    name: 'bernoulli-table',
    maps: 'a Bernoulli whose probability depends on discrete latents: a table factor over the parents and the node',
    families: ['bernoulli'],
    requires: ['table'],
    matches: (_cx, node, rows) => node.dist.family === 'bernoulli' && rows.parents.length > 0,
    map: (cx, node, rows) => observedBernoulliTable(cx, node, rows.values, rows.parents),
  },
  {
    name: 'elements',
    maps: 'every other observed node, element by element, by the table of its family',
    requires: [],
    map: (cx, node, rows) => {
      const n = prod(cx.shapeOf(node.dims))
      const vars: (string | null)[] = []
      for (let i = 0; i < n; i++) {
        const y = rows.values[i] as number
        if (Number.isNaN(y)) {
          vars.push(null)
          continue
        }
        const name = cx.elementName(node.name, node.dims, i)
        vars.push(
          mapFirst(observedElementMappings, cx, 'observed element', node, {
            fam: node.dist.family,
            name,
            y,
            i,
            index: node.dims.length === 0 ? undefined : i,
          }),
        )
      }
      cx.layout(node, 'observed', undefined, node.dims.length === 0 ? 'scalar' : 'observed', vars, false)
    },
  },
]

/**
 * What a whole observed node needs before its rows are mapped: the policy for missing values (only `ignore`), no subsampling by a
 * batch, and its observed values as numbers. Then the rows are mapped by `observedRowMappings`.
 */
function observedRows(cx: Context, node: ObservedNode): void {
  const info = cx.bound.missing[node.name]
  if (info !== undefined && info.count > 0 && info.policy !== 'ignore')
    throw new Refusal(
      'data',
      `'${node.name}' has ${info.count} missing value(s) with the policy '${info.policy}': only 'ignore' is mapped (the entries drop out of the graph)`,
    )
  if (cx.bound.batch !== undefined && node.dims.includes(cx.bound.batch.dim))
    throw new Refusal(
      'structure',
      `'${node.name}' is subsampled by the batch option: a factor graph is built over all the rows`,
    )
  const values = cx
    .over(cx.fromNamed(evalNamedValue(cx, node)), node.dims)
    .map((c) => (c as Cell & { kind: 'aff' }).aff.offset)
  mapFirst(observedRowMappings, cx, 'observed rows', node, { values, parents: discreteParents(cx, node) })
}

// ----- the elements of an observed node -----

/** A mapping of an element, for the families it serves (`families` is the test, so an entry needs no `matches` for the family). */
function element(
  entry: Omit<Mapping<[ObservedNode, Element], string | null>, 'families'> & { readonly families: readonly string[] },
): Mapping<[ObservedNode, Element], string | null> {
  const { matches, families } = entry
  return {
    ...entry,
    matches: (cx, node, el) => families.includes(el.fam) && (matches === undefined || matches(cx, node, el)),
  }
}

/** Whether the scale of a normal element is random. */
const randomScale = (cx: Context, node: ObservedNode): ExprSpec | undefined => {
  const scale = node.dist.params['scale']
  return scale !== undefined && !cx.isConst(scale) ? scale : undefined
}

/** The factor of an unmapped parameter form: the entries that refuse after a numeric mapping that did not take the node. */
const unmappedParameters = (node: ObservedNode, fam: string, what = ''): never => {
  throw new Refusal(
    'operator',
    `the parameters of '${node.name}' (${fam}) are not in a form with a message rule${what}`,
  )
}

/**
 * The mappings of one element of an observed node, in priority order; the families are independent of each other (the order across
 * families does not matter), the order within a family does. The last entry refuses a family with no mapping.
 */
export const observedElementMappings: MappingTable<[ObservedNode, Element], string | null> = [
  element({
    name: 'normal-switch-log-scale',
    maps: 'a Gaussian whose log scale (and mean) is chosen by one discrete variable among random forms: one gate of gaussian-scale branches',
    order:
      'first for the normal: it takes only a scale exp(s) with s a lookup of a discrete variable; NOT_MINE sends the rest down the chain',
    families: ['normal'],
    requires: ['gate', 'gaussian-scale'],
    map: (cx, node, el) => observedSwitchLogScale(cx, node, el.y, el.i, el.index),
  }),
  element({
    name: 'normal-switched',
    maps: 'a Gaussian whose mean or scale is a lookup by a discrete variable: a table over it (known means) or a gate of Gaussian likelihoods',
    order: 'before the random-scale forms: a scale chosen by a discrete variable is not a random scale',
    families: ['normal'],
    requires: [
      'bernoulli',
      'discrete',
      'gate',
      'gaussian',
      'gaussian-gamma-precision',
      'gaussian-scale',
      'likelihood-site',
      'prior',
      'table',
    ],
    map: (cx, node, el) => {
      const switched = switchedCells(cx, node, el.i)
      if (switched === undefined) return NOT_MINE
      return observedSwitch(cx, node, el.name, el.y, el.index, switched.loc, switched.scale)
    },
  }),
  element({
    name: 'normal-log-scale',
    maps: 'a Gaussian whose scale is exp of an affine form of Gaussian variables (stochastic volatility): a gaussian-log-scale factor, the mean known',
    order:
      'before the precision idiom and the random scale: NOT_MINE (a random mean, or a log that is not a Gaussian form) falls to them',
    families: ['normal'],
    requires: ['gaussian-log-scale'],
    matches: (cx, node) => {
      const scale = randomScale(cx, node)
      return scale !== undefined && expArgument(unwrap(cx.nodes, scale)) !== undefined
    },
    map: (cx, node, el) => observedLogScale(cx, node, el.name, el.y, el.i, el.index),
  }),
  element({
    name: 'normal-random-scale',
    maps: 'a Gaussian whose scale is a positive random value that is not the precision idiom (exp of a Gaussian, a half-normal sd): a gaussian-scale factor over the log scale',
    order: 'before the precision idiom: it takes the scales that are not 1 / sqrt(tau)',
    families: ['normal'],
    requires: ['gaussian-scale'],
    matches: (cx, node) => {
      const scale = randomScale(cx, node)
      return scale !== undefined && precisionExpr(cx.idiom, scale) === undefined
    },
    map: (cx, node, el) =>
      observedRandomScale(cx, node, el.name, el.y, el.i, el.index, randomScale(cx, node) as ExprSpec),
  }),
  element({
    name: 'normal-random-precision',
    maps: 'a Gaussian whose scale is the precision idiom 1 / sqrt(tau) of a gamma latent: a gaussian-gamma-precision factor (the mean a variable or known)',
    families: ['normal'],
    requires: ['gaussian-gamma-precision'],
    matches: (cx, node) => randomScale(cx, node) !== undefined,
    map: (cx, node, el) => observedRandomPrecision(cx, node, el.name, el.y, el.i, el.index),
  }),
  element({
    name: 'normal',
    maps: 'a Gaussian with a known scale: a gaussian factor on the mean (a regression row on a weight vector, or a variable)',
    order: 'last for the normal: the known-scale form',
    families: ['normal'],
    requires: ['gaussian', 'linear-regression-row'],
    map: (cx, node, el) => observedGaussian(cx, node, el.name, el.y, el.i, el.index, 'loc', 'scale'),
  }),
  element({
    name: 'heavy-tailed',
    maps: 'a Student-t, Laplace or outlier-mixture observation: the scale-mixture factors of the engine on the mean',
    families: ['studentT', 'laplace', 'normalOutliers'],
    requires: [
      'gate',
      'gaussian',
      'gaussian-gamma-precision',
      'gaussian-scale',
      'likelihood-site',
      'linear-regression-row',
      'prior',
    ],
    map: (cx, node, el) => observedHeavy(cx, node, el.fam, el.y, el.i, el.index),
  }),
  element({
    name: 'beta',
    maps: 'a beta observation with random shapes: a likelihood-pair factor',
    families: ['beta'],
    requires: ['likelihood-pair'],
    map: (cx, node, el) => observedPair(cx, node, 'beta-shapes', el.y, el.i, el.index),
  }),
  element({
    name: 'ordered-numeric',
    maps: 'an ordered probit or logistic, or an asymmetric Laplace (quantile regression): a likelihood-site factor on the predictor',
    families: ['orderedProbit', 'orderedLogistic', 'asymmetricLaplace'],
    requires: ['likelihood-pair', 'likelihood-site', 'likelihood-row'],
    map: (cx, node, el) => observedNumeric(cx, node, el.fam, el.y, el.i, el.index),
  }),
  element({
    name: 'ordered-unmapped',
    maps: 'the parameters of an ordered or quantile likelihood that are not a form with a message rule are refused',
    families: ['orderedProbit', 'orderedLogistic', 'asymmetricLaplace'],
    requires: [],
    map: (_cx, node, el) => unmappedParameters(node, el.fam),
  }),
  element({
    name: 'log-normal',
    maps: 'a log-normal observation: log y is Gaussian around the predictor, the exact Gaussian rule on the log of the outcome',
    families: ['logNormal'],
    requires: ['gaussian', 'linear-regression-row'],
    map: (cx, node, el) => {
      if (!(el.y > 0)) throw new Refusal('data', `'${node.name}' is log-normal: the outcome ${el.y} must be positive`)
      return observedGaussian(cx, node, el.name, Math.log(el.y), el.i, el.index, 'mu', 'sigma')
    },
  }),
  element({
    name: 'bernoulli',
    maps: 'a Bernoulli observation: a probit or logistic link of an affine form, or a beta probability',
    families: ['bernoulli'],
    requires: ['bernoulli', 'logistic', 'probit'],
    map: (cx, node, el) => observedBernoulli(cx, node, el.name, el.y, el.i, el.index),
  }),
  element({
    name: 'binomial-poisson-numeric',
    maps: 'a Poisson with a log link, a binomial with a logit or probit link: the numeric likelihood of the registry on the predictor',
    order:
      'before the conjugate rule: the numeric form is tried first, NOT_MINE when the rate or probability is a conjugate latent',
    families: ['binomial', 'poisson'],
    requires: ['gate', 'likelihood-row', 'likelihood-site'],
    map: (cx, node, el) => observedNumeric(cx, node, el.fam, el.y, el.i, el.index),
  }),
  element({
    name: 'binomial-poisson-conjugate',
    maps: 'a Poisson with a gamma rate, a binomial with a beta probability: the conjugate observation',
    families: ['binomial', 'poisson'],
    requires: ['binomial', 'gate', 'poisson'],
    map: (cx, node, el) => observedConjugate(cx, node, el.fam, el.name, el.y, el.i),
  }),
  element({
    name: 'multiclass-logits',
    maps: 'a categorical with logits: a multiclass-site factor on the predictors of the classes',
    order: 'before the conjugate categorical: logits are the multiclass form',
    families: ['categorical'],
    requires: ['multiclass-site'],
    matches: (_cx, node) => node.dist.params['logits'] !== undefined,
    map: (cx, node, el) => observedMulticlass(cx, node, el.y, el.i, el.index, 'logits', 'logit'),
  }),
  element({
    name: 'categorical-conjugate',
    maps: 'a categorical with Dirichlet probabilities: a discrete factor on the Dirichlet variable, or a gate over rows chosen by a discrete latent',
    families: ['categorical'],
    requires: ['discrete', 'gate'],
    map: (cx, node, el) => observedConjugate(cx, node, el.fam, el.name, el.y, el.i),
  }),
  element({
    name: 'categorical-probit',
    maps: 'a categorical with probit scores: a multiclass-site factor',
    families: ['categoricalProbit'],
    requires: ['multiclass-site'],
    map: (cx, node, el) => observedMulticlass(cx, node, el.y, el.i, el.index, 'scores', 'probit'),
  }),
  element({
    name: 'zero-truncated-poisson-numeric',
    maps: 'a zero-truncated Poisson with a log link: the numeric likelihood of the registry',
    families: ['zeroTruncatedPoisson'],
    requires: ['likelihood-row', 'likelihood-site'],
    map: (cx, node, el) => observedNumeric(cx, node, el.fam, el.y, el.i, el.index),
  }),
  element({
    name: 'zero-truncated-poisson-unmapped',
    maps: 'a zero-truncated Poisson whose rate is not exposure * exp(affine) is refused',
    families: ['zeroTruncatedPoisson'],
    requires: [],
    map: (_cx, node) => {
      throw new Refusal(
        'operator',
        `the rate of '${node.name}' must be \`exposure * exp(affine)\` for the zero-truncated Poisson`,
      )
    },
  }),
  element({
    name: 'hurdle-poisson',
    maps: 'a hurdle Poisson: a Bernoulli for zero and a zero-truncated Poisson for the rest',
    families: ['hurdlePoisson'],
    requires: ['bernoulli', 'likelihood-site', 'likelihood-row', 'prior'],
    map: (cx, node, el) => observedHurdle(cx, node, el.y, el.i, el.index),
  }),
  element({
    name: 'mixture-weight',
    maps: 'a pair of Gaussian densities weighted by a beta latent (the clutter problem): a mixture-weight factor',
    families: ['normalPair'],
    requires: ['mixture-weight'],
    map: (cx, node, el) => observedMixtureWeight(cx, node, el.name, el.y, el.i),
  }),
  element({
    name: 'gamma-rate',
    maps: 'an exponential or gamma observation whose rate is a gamma latent: a gamma-observed factor',
    families: ['exponential', 'gamma'],
    requires: ['gamma-observed'],
    map: (cx, node, el) => observedGammaRate(cx, node, el.fam, el.name, el.y, el.i),
  }),
  element({
    name: 'geometric',
    maps: 'a geometric observation with a beta probability: a negative-binomial-beta factor (the failures before a success)',
    families: ['geometric'],
    requires: ['negative-binomial-beta'],
    map: (cx, node, el) => observedFailures(cx, node, el.fam, el.name, el.y, el.i),
  }),
  element({
    name: 'negative-binomial-weibull-numeric',
    maps: 'a negative binomial with a log link (a known or random dispersion), a Weibull with a log-linear scale: the numeric likelihood of the registry',
    order: 'before the beta-conjugate failures of the negative binomial',
    families: ['negativeBinomial', 'weibull'],
    requires: ['likelihood-pair', 'likelihood-row', 'likelihood-site', 'negative-binomial-dispersion'],
    map: (cx, node, el) => observedNumeric(cx, node, el.fam, el.y, el.i, el.index),
  }),
  element({
    name: 'negative-binomial-failures',
    maps: 'a negative binomial with a beta probability: a negative-binomial-beta factor',
    families: ['negativeBinomial'],
    requires: ['negative-binomial-beta'],
    map: (cx, node, el) => observedFailures(cx, node, el.fam, el.name, el.y, el.i),
  }),
  element({
    name: 'weibull-unmapped',
    maps: 'the parameters of a Weibull that are not a form with a message rule are refused',
    families: ['weibull'],
    requires: [],
    map: (_cx, node, el) => unmappedParameters(node, el.fam, ' (see the log-link idioms of its observedNumeric)'),
  }),
  {
    name: 'family-unmapped',
    maps: 'a family with no likelihood mapping is refused, listing the likelihoods mapped',
    order: 'last: takes every element that no entry above took',
    requires: [],
    map: (_cx, node, el) => {
      throw new Refusal(
        'family',
        `'${node.name}' is observed as ${el.fam}: the likelihoods mapped are normal, logNormal, studentT, laplace, asymmetricLaplace, orderedProbit, orderedLogistic, categoricalProbit, normalOutliers, bernoulli, binomial, poisson, normalPair, exponential, gamma, geometric, negativeBinomial, weibull and categorical`,
      )
    },
  },
]
