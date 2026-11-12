/**
 * The vocabulary of an engine run: the slots of its settings (`EngineSettings`, the improper policy, the start messages),
 * its state (a message on every edge and where the run is in its sweep), the counts of the improper-message repair and the
 * snapshot of a run. Types only; nothing computes here.
 */

import type { Json } from '../../base/index.ts'
import type { Eta } from '../core/types.ts'
import type { Projection } from './projections.ts'
import type { CustomOrder, Granularity, ScheduleKind } from './schedules.ts'

/** What to do when an update would make a proper belief improper (a negative precision: the usual failure of EP). */
export type ImproperPolicy = 'skip' | 'damp' | 'reset' | 'error'

/** A message to start from instead of the flat one. */
export interface InitialMessage {
  /** The factor's index. */
  readonly factor: number
  /** The port. */
  readonly port: number
  readonly eta: Eta
}

/** The slots of the engine. */
export interface EngineSettings {
  /** The message rule: moment matching, expected log, sum-product. */
  readonly projection: Projection
  /** The order of a sweep. */
  readonly schedule: ScheduleKind
  /** The order of a `custom` schedule. */
  readonly order?: CustomOrder
  /** How much of a sweep a step does. */
  readonly granularity: Granularity
  /** Damping in [0, 1): the new message is `(1 - damping) new + damping old` in natural parameters. */
  readonly damping: number
  /** The repair of an update that makes a proper belief improper. */
  readonly improper: ImproperPolicy
  /** Stop when no belief moves by more than this in a sweep (in the units of the family's change measure). */
  readonly tol: number
  /** Stop, uncompleted, after this many sweeps. */
  readonly maxSweeps: number
  /** Messages to start from. */
  readonly initial?: readonly InitialMessage[]
  /**
   * The first sweeps of a run (and of each restart) are damped by at least a half (`WARMUP_DAMPING`), then the run goes on at its
   * own damping (default 0: none). A start from flat messages commits every row to a side at its first undamped updates; where the
   * labels of a mixture are exchangeable that decides which fixed point the run falls into (one blob, the symmetric one), and a few
   * damped sweeps let the components separate first.
   */
  readonly warmup?: number
  /**
   * How many times a run that stalls (or diverges) is restarted from its start messages with a stronger damping before it is
   * reported as stalled (default 0: report it at once). The restarts are damped 0.5, 0.8 and 0.9, those stronger than the run's own (a
   * run damped by 0.5 restarts at 0.8, then 0.9). A run already damped by 0.9 or more is not restarted, since a restart would repeat it
   * exactly: it is judged as the restart would be (see `mpVerdict`).
   */
  readonly restarts?: number
}

/** How often the improper-message repair acted. */
export interface ImproperCounts {
  /** Updates that would have made a belief improper, or failed on an improper quantity. */
  readonly events: number
  /** Updates dropped (the old message kept). */
  readonly skipped: number
  /** Updates shortened by halving the step until the belief stayed proper. */
  readonly damped: number
  /** Messages reset to flat. */
  readonly reset: number
}

/** The state of a run: a message on every edge, and where the run is in its sweep. */
export interface MpState {
  /** The message of every edge (factor to variable), natural parameters. */
  readonly messages: readonly (readonly number[])[]
  /** The beliefs at the start of the sweep, for the change over it. */
  readonly start: readonly (readonly number[])[]
  /** Flooding: the messages at the start of the sweep that every message of it is computed from. */
  readonly frozen: readonly (readonly number[])[] | null
  /** Completed sweeps. */
  readonly sweeps: number
  /** The step within the sweep (the group of edges to send next); 0 between sweeps. */
  readonly cursor: number
  /** Messages sent so far. */
  readonly updates: number
  /** The largest change of a belief over the last completed sweep (Infinity before the first). */
  readonly change: number
  readonly improper: ImproperCounts
  /** The edges sent by the last step. */
  readonly sent: readonly number[]
  /** The residual schedule: the message each edge would send, and how far it is from the one it has. */
  readonly candidates: readonly (readonly number[] | null)[]
  readonly residuals: readonly number[]
  /** The updates of the last completed sweep that could not be computed or were dropped by the improper policy: the messages that did not move because they could not be recomputed. */
  readonly unresolved: number
  /** The same count over the sweep in progress. */
  readonly pending: number
  /** The damping in force: the setting, raised by each restart. */
  readonly damping: number
  /** The automatic restarts so far (see `EngineSettings.restarts`). */
  readonly restarts: number
  /** The sweeps completed when the last restart began (0 for a run that never restarted): the sweep cap counts from there. */
  readonly restartAt: number
}

/** A picture of a run at one moment: where it is, and what it currently believes of every variable, by variable id. */
export interface MpSnapshot {
  readonly sweeps: number
  readonly updates: number
  readonly change: number
  readonly improper: ImproperCounts
  /** The updates of the last completed sweep that could not be computed (see {@link MpState}); absent from the snapshots of runs that do not count them. */
  readonly unresolved?: number
  /** The current marginal of every variable by id (see {@link marginalsOf}). */
  readonly marginals: Readonly<Record<string, Readonly<Record<string, Json>>>>
}
