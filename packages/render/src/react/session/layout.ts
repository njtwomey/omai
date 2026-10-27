/**
 * The vocabulary of session layouts: the kinds of panel a page declares over a session's tape. What the tape logs and
 * what a page shows are separate; the layout alone decides what is drawn, and every panel reads one shared step cursor.
 */

import type { Json } from 'mdml-engine/base'

/**
 * One panel of a layout. Panels of the tape (`curves`, `series`) scrub the cursor on click; panels of a step (`rows`,
 * `probe`, `tensor`, `trajectory`) follow it.
 */
export type Panel =
  /** The training rows with the probe overlay at the viewed step (sessions on a dataset only). */
  | {
      readonly kind: 'rows'
      /**
       * `assignments` and centres (`model/assignments`, `model/centres`); `components` those with each component's 1σ
       * and 2σ ellipses (`model/params` with covariances); `fit` the 1-D fit (`model/predictions`); `band` the 1-D fit
       * with its predictive band (`model/predictive`: mean, lower, upper per row); `surface` the decision surface
       * (`model/decision-surface`); `experts` a 1-D mixture of experts: rows coloured by their most responsible expert
       * (`model/assignments`), one thin line per expert in its colour (`model/experts`: `{ lines }`, expert by row) and
       * the mixture's own line, bolder (`model/predictions`).
       */
      readonly overlay?: 'assignments' | 'components' | 'fit' | 'band' | 'surface' | 'experts'
      readonly title?: string
    }
  /** A derived path at the viewed step: an exposition record is drawn by its view, anything else as JSON. */
  | { readonly kind: 'probe'; readonly path: string; readonly args?: Json; readonly title?: string }
  /** Scalar paths of the tape as curves on one chart, against the step. */
  | { readonly kind: 'curves'; readonly paths: readonly string[]; readonly title?: string }
  /**
   * Each element of a tensor path as a curve, against the step or against a scalar path (`x`, the time an equation has
   * reached, say): the paths of a process, the weights of a mixture, the components of a state. Up to `max` elements
   * (default 32), named by `names` or their index.
   */
  | {
      readonly kind: 'series'
      readonly path: string
      readonly x?: string
      readonly names?: readonly string[]
      readonly max?: number
      readonly title?: string
    }
  /**
   * Two elements (`dims`, default the first two) of tensor paths as paths in the plane up to the viewed step, each
   * ending in its current point: a chain's draws, an equation's phase portrait, a filter's estimate against the truth.
   * `over` draws a probed surface (a `decisionSurface` record with `values`: a target density, an objective) as
   * contours beneath; `ellipse` names a covariance tensor path drawn as the 1σ ellipse at the first path's current
   * point; `trail` keeps only the last so many points; `points` draws the paths as points instead of lines.
   */
  | {
      readonly kind: 'trajectory'
      readonly paths: readonly string[]
      readonly dims?: readonly [number, number]
      readonly over?: { readonly path: string; readonly args?: Json }
      readonly ellipse?: string
      readonly trail?: number
      readonly points?: boolean
      readonly title?: string
    }
  /** A tensor path at the viewed step as a heatmap (a matrix being factorised, a transition matrix, an image). */
  | { readonly kind: 'tensor'; readonly path: string; readonly title?: string }
  /** One of the method's projections of the tape so far; hidden while it has nothing to project. */
  | { readonly kind: 'projection'; readonly name: string; readonly title?: string }
  /** The generic view of everything the tape holds, for debugging. */
  | { readonly kind: 'tape' }
