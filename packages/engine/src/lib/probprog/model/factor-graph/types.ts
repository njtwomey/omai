/**
 * What a compilation of a model to a factor graph reports: the issues (the reasons a node could not be mapped), the layout of
 * each site in the graph and the compilation record `toFactorGraph` returns and `factorGraphPosterior` reads back.
 */

import type { FactorGraphSpec, FamilyKind } from '../../../message-passing/index.ts'

/** Why a part of a model could not be put on a factor graph: the kind says what to look at. */
export type FactorIssueKind =
  /** A `custom` call: a function the graph cannot see through. */
  | 'opaque'
  /** A distribution family with no message family in the engine. */
  | 'family'
  /** An operator with no message rule on a random value, or an expression that is not affine. */
  | 'operator'
  /** A shape of model the first version does not expand (event axes, plated discrete tables). */
  | 'structure'
  /** Data that is not bound, a missing size, missing values. */
  | 'data'
  /** A constraint beyond the family's own support. */
  | 'constraint'
  /** A kind of node with no counterpart on a factor graph (`param`, a `factor` term) or one that depends on a node that failed. */
  | 'node'

/** One thing a model has that the compiler cannot map, with the node it is on and the reason in words. */
export interface FactorIssue {
  readonly node: string
  readonly kind: FactorIssueKind
  readonly reason: string
}

/** How a site of the spec is laid out in the graph, so results can be read back in the spec's shapes. */
export interface SiteLayout {
  readonly site: string
  readonly kind: 'latent' | 'observed' | 'derived'
  readonly dims: readonly string[]
  /** The sizes of `dims`. */
  readonly shape: readonly number[]
  /** The engine family of the site's variables. */
  readonly family: FamilyKind | undefined
  /**
   * `scalar`: one variable named `site`. `elements`: a variable per element, `site[i]`, in row-major order of `dims`.
   * `packed`: one vector variable named `site` over the elements. `observed`: the variables hold the data (no marginal).
   * `constant`: a derived value with no random part, so no variable.
   */
  readonly layout: 'scalar' | 'elements' | 'packed' | 'observed' | 'constant'
  /** The variable of each element in row-major order (`null` where an element has none: a constant or a missing value). */
  readonly variables: readonly (string | null)[]
  /** Whether the site is one of a posterior's sites (latents and recorded derived values). */
  readonly posterior: boolean
  /**
   * A site that is the reciprocal of the gamma variable that carries it (an inverse-gamma variance whose precision is the
   * variable): the posterior reads it as an inverse gamma.
   */
  readonly reciprocal?: boolean
  /**
   * For a site read from a normal-gamma variable shared with another site (the mean and the precision of a Gaussian, the
   * conjugate pair): which part of the joint it is, `mean` (a Student t marginal) or `precision` (a gamma marginal).
   */
  readonly part?: 'mean' | 'precision'
  /** For a beta variable that carries a correlation `rho = 2 u - 1`: the posterior is read as the correlation, not as `u`. */
  readonly correlation?: boolean
}

/** What a compilation to a factor graph reports. */
export interface FactorGraphCompilation {
  readonly name: string
  /** The graph, or `undefined` when something could not be mapped (see `issues`). */
  readonly graph: FactorGraphSpec | undefined
  readonly issues: readonly FactorIssue[]
  /** The nodes message passing cannot see through (`analyseSpec`'s `opaque`: a `custom` call or a family with no exponential-family form). */
  readonly opaque: readonly string[]
  readonly sites: readonly SiteLayout[]
  /** The auxiliary variables the expansion introduced (`y#0[3]`). */
  readonly auxiliary: readonly string[]
  /** The observed sites compiled as one plated factor over all their rows (no variable or factor per row), by site name. */
  readonly plated: readonly string[]
  /** What the expansion made: the plates that were unrolled and the sizes of the result. */
  readonly expanded: {
    readonly plates: readonly { readonly dim: string; readonly size: number }[]
    readonly variables: number
    readonly factors: number
  }
}
