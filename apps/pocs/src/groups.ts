import type { PocGroup } from './poc.ts'

/**
 * The groups of proofs of concept, in the order the index shows them. A poc joins one with `group: '<slug>'` in its
 * `poc.ts`; a group with no pocs is not shown. Add a group here when a set of pocs belongs together.
 */
export const GROUPS: readonly PocGroup[] = [
  {
    slug: 'bayesian-examples',
    title: 'Bayesian examples',
    summary:
      'Canonical Bayesian models, from a coin to hierarchical and latent-variable models. Each page is one experiment: choose the data, the model and one inference method, see what it infers, and check it against a reference.',
  },
  {
    slug: 'probabilistic-programming',
    title: 'Probabilistic programming (earlier pages, parked)',
    summary:
      'The first pages on models written as programs or graphs and the ways of inferring from them. Parked while the Bayesian examples are built.',
  },
  {
    slug: 'structured-prediction',
    title: 'Structured prediction',
    summary: 'Labelling sequences, trees and graphs: conditional random fields, their features, solvers and decoders.',
  },
  {
    slug: 'models',
    title: 'Models and datasets',
    summary: 'Additive models, Kolmogorov-Arnold networks and the real datasets the library ships.',
  },
  {
    slug: 'graphics-and-fields',
    title: 'Neural fields and graphics',
    summary:
      'Rendering with learned and classical representations: radiance fields, splats, implicit surfaces, ray tracing.',
  },
] as const
