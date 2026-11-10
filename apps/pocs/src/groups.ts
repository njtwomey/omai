import type { PocGroup } from './poc.ts'

/**
 * The groups of proofs of concept, in the order the index shows them. A poc joins one with `group: '<slug>'` in its
 * `poc.ts`; a group with no pocs is not shown. Add a group here when a set of pocs belongs together.
 */
export const GROUPS: readonly PocGroup[] = [
  {
    slug: 'probabilistic-programming',
    title: 'Probabilistic programming and inference',
    summary:
      'Models written as programs or graphs and the ways of inferring from them: exact and sequential Bayesian updates, message passing, variational inference and sampling.',
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
