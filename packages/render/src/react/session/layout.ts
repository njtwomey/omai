/**
 * The vocabulary of session layouts: the kinds of panel a page declares over a session's tape. What the tape logs and
 * what a page shows are separate; the layout alone decides what is drawn, and every panel reads one shared step cursor.
 * Over a comparison of several variants the same layout serves: panels of the tape (`curves`, `sweep`) overlay the
 * variants, one colour each, and the rest are drawn once per variant, side by side.
 */

import type { Json } from 'mdml-engine/base'
import type { MatrixMark } from '../../views/matrix-marks.ts'

/**
 * Where a channel of the rows panel is read at the viewed step: a derived path (probed with `args`) or a tensor path of
 * the tape (the latest full value at or before the step).
 */
export interface RowSource {
  readonly path: string
  readonly args?: Json
}

/** A line the estimates are judged against: a constant (`value`) or a path of the tape (`path`), drawn dashed. */
export interface TruthLine {
  readonly value?: number
  readonly path?: string
  readonly label?: string
}

/** A band around a scalar path, between two other paths (a metric with its lower and upper bounds). */
export interface BandSpec {
  readonly path: string
  readonly lower: string
  readonly upper: string
  readonly name?: string
}

/** A reference slope for a log-log chart, drawn through the first curve's first point (`through` overrides). */
export interface ReferenceSlope {
  readonly slope: number
  readonly label?: string
  readonly through?: readonly [number, number]
}

/** Events of the tape as vertical marks: the steps at which a scalar path was logged with a non-zero value. */
export interface EventMarks {
  readonly path: string
  readonly label?: string
}

/**
 * One panel of a layout. Panels of the tape (`curves`, `series`, `sweep`) scrub the cursor on click; panels of a step
 * (`rows`, `probe`, `tensor`, `trajectory`, `field`, `strip`, `table`, `images`, `graph`, `points`, `scene`) follow it.
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
      /**
       * A number per row as colour along a ramp, with a colour bar (`diverging`: a ramp centred on zero, for weights,
       * attributions and residuals; `range` fixes the ends; `label` titles the bar). The class shows as marker shape.
       */
      readonly values?: RowSource & {
        readonly label?: string
        readonly diverging?: boolean
        readonly range?: readonly [number, number]
      }
      /** A number per row as marker size (`range` in pixels, default 4 to 16). */
      readonly size?: RowSource & { readonly range?: readonly [number, number] }
      /**
       * A category per row (`null` for none) or `{ sets: { name: [rows] } }`, each category a hollow marker shape named
       * in the legend (`names` renames): labelled against unlabelled, flagged, chosen at step k, a witness.
       */
      readonly marks?: RowSource & { readonly names?: Readonly<Record<string, string>> }
      /** Regions of the plane: boxes (`{ lower, upper }`, open sides `null`) or ellipses (`{ centre, radii, angle }`). */
      readonly regions?: RowSource
      /**
       * One chosen row (`{ row, neighbours?, path?, radius?, point? }`) ringed, with thin lines to its neighbours, a
       * bold path through rows, a circle of a radius around it and a related point joined by a dashed line.
       */
      readonly focus?: RowSource
      readonly title?: string
    }
  /** A derived path at the viewed step: an exposition record is drawn by its view, anything else as JSON. */
  | { readonly kind: 'probe'; readonly path: string; readonly args?: Json; readonly title?: string }
  /** Scalar paths of the tape as curves on one chart, against the step. */
  | {
      readonly kind: 'curves'
      readonly paths: readonly string[]
      /**
       * The variants of a comparison are repeats of one thing: each one a faint curve, the mean across them in bold
       * (one colour per path instead of per variant).
       */
      readonly runs?: boolean
      /** Bands (and, nested, fans) between a lower and an upper path, in the colour of the path's curve. */
      readonly bands?: readonly BandSpec[]
      /** A dashed line to compare against: a constant or a path. */
      readonly truth?: TruthLine
      /** Logarithmic axes (non-positive points are left out). */
      readonly log?: 'x' | 'y' | 'xy'
      /** Reference slopes on log-log axes. */
      readonly slopes?: readonly ReferenceSlope[]
      /** Vertical marks where scalar paths fired: planted changes, alarms, restarts. */
      readonly events?: readonly EventMarks[]
      readonly title?: string
    }
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
  /**
   * A sweep: a metric logged once per trial of a study (`path`) against the knob each trial set (`over`, a scalar path
   * logged at the same step; default the step itself), each point a trial, those after the viewed step faint and the
   * viewed one ringed; a click scrubs to the nearest trial's step.
   */
  | {
      readonly kind: 'sweep'
      readonly path: string
      readonly over?: string
      /** Logarithmic axes (non-positive trials are left out). */
      readonly log?: 'x' | 'y' | 'xy'
      /** A dashed horizontal line to compare against (`value`); with `intervals`, the truth: a value or a path. */
      readonly truth?: TruthLine
      /** Reference slopes on log-log axes. */
      readonly slopes?: readonly ReferenceSlope[]
      /**
       * An interval strip instead: `path` is each trial's estimate and the bounds are two paths logged at the same
       * steps; every trial a horizontal bar, the misses of `truth` (a constant or a path) in red.
       */
      readonly intervals?: { readonly lower: string; readonly upper: string }
      readonly title?: string
    }
  /**
   * A tensor path at the viewed step as a heatmap (a matrix being factorised, a transition matrix, an image). Given
   * `rowLabels`, `colLabels` or marks it is drawn as labelled cells with the values written in: `marks` are fixed
   * (a boundary between users and items), `marksPath` is a probe at the viewed step returning marks (the cell just
   * updated, the observed entries): a cell or a row or a column outlined, a dashed boundary between two rows or two
   * columns.
   */
  | {
      readonly kind: 'tensor'
      readonly path: string
      readonly rowLabels?: readonly string[]
      readonly colLabels?: readonly string[]
      readonly marks?: readonly MatrixMark[]
      readonly marksPath?: string
      readonly marksArgs?: Json
      readonly title?: string
    }
  /**
   * A vector field probed at the viewed step (a `vectorField` record: a flow, a score field, a phase portrait with
   * nullclines and classified fixed points) drawn as arrows or streamlines, and, with `trajectory`, tensor paths of the
   * tape over it up to the viewed step, each ending in its current point.
   */
  | {
      readonly kind: 'field'
      readonly path: string
      readonly args?: Json
      readonly style?: 'arrows' | 'streamlines' | 'both'
      readonly trajectory?: {
        readonly paths: readonly string[]
        readonly dims?: readonly [number, number]
        readonly trail?: number
        readonly points?: boolean
      }
      readonly title?: string
    }
  /**
   * A position-aligned strip probed at the viewed step (a `strip` record): tokens on x, a row of cells per track beneath
   * them (probabilities, an attention row, gates, the states of a trellis), tags, the chosen path and marks.
   */
  | { readonly kind: 'strip'; readonly path: string; readonly args?: Json; readonly title?: string }
  /**
   * A small table probed at the viewed step (a `markedTable` record): cells tinted by status, the decision outlined.
   */
  | { readonly kind: 'table'; readonly path: string; readonly args?: Json; readonly title?: string }
  /**
   * Many small images probed at the viewed step (an `imageSteps` record; the latest step at or before the viewed one):
   * a row per group with its name, and a caption under each image when the step has them.
   */
  | {
      readonly kind: 'images'
      readonly path: string
      readonly args?: Json
      /** The most images drawn per group (default 12). */
      readonly columns?: number
      readonly title?: string
    }
  /**
   * A graph probed at the viewed step (a `structureSteps` record): the same nodes and edges, each node's colour, size,
   * label and position and each edge's width, label and dash read from the latest recorded step at or before the viewed
   * one.
   */
  | {
      readonly kind: 'graph'
      readonly path: string
      readonly args?: Json
      readonly layout?: 'force' | 'circular'
      readonly title?: string
    }
  /**
   * A point set probed at the viewed step (a `pointsSteps` record): the points of every step up to it, or the set at it,
   * on axes that hold still over the whole run; in three dimensions a cloud to orbit.
   */
  | { readonly kind: 'points'; readonly path: string; readonly args?: Json; readonly title?: string }
  /**
   * A scene probed at the viewed step (a `sceneSteps` record): the latest recorded scene at or before it, with its
   * caption (agents on a plane, a plant and its horizon, a drawing being built).
   */
  | { readonly kind: 'scene'; readonly path: string; readonly args?: Json; readonly title?: string }
  /** One of the method's projections of the tape so far; hidden while it has nothing to project. */
  | { readonly kind: 'projection'; readonly name: string; readonly title?: string }
  /** The generic view of everything the tape holds, for debugging. */
  | { readonly kind: 'tape' }
