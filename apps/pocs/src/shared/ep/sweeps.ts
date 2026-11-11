/**
 * How an EP program tells its page how far it is. A program's `infer` takes optional `EpHooks`: it reports each sweep
 * (`onSweep`), asks whether to go on (`stopped`) and hands the thread back to the page between sweeps (`yieldToPage`);
 * called without them it runs straight through, as the tests do. `epHooks` makes them from the host of a run, with the
 * stage's name, so the bar shows 'sweep 41 of 300' and the change against the tolerance.
 */

import { type AnytimeResult, runAnytimeAsync } from 'omai-engine/probprog/inference'
import type { Config, Session } from 'omai-engine/trace'
import type { RunHost } from '../run.ts'

/** The hooks of a program that steps: all optional. */
export interface EpHooks {
  /** Called after each sweep (or step) with how many are done, the cap, and, for EP, the change and the tolerance. */
  readonly onSweep?: (sweep: {
    readonly sweep: number
    readonly cap: number
    readonly change?: number
    readonly tol?: number
  }) => void
  /** Whether to stop where it is. */
  readonly stopped?: () => boolean
  /** Hands the thread back to the page. */
  readonly yieldToPage?: () => Promise<void>
}

const exp = (v: number): string => (Number.isFinite(v) ? v.toExponential(1) : 'not yet')

/**
 * The hooks of a stage of a run: its sweeps go to the bar (done over the cap, and the change against the tolerance as the
 * second line), its stop is the host's and its slices hand the thread back through the host.
 *
 * @param host - The host of the run.
 * @param stage - What is running ('EP', 'Gibbs reference').
 * @param unit - The word for one step ('sweep', 'iteration', 'seed').
 * @returns The hooks, to pass to a program's `infer`.
 */
export function epHooks(
  host: Pick<RunHost, 'report' | 'stopped' | 'yieldToPage'>,
  stage: string,
  unit = 'sweep',
): EpHooks {
  return {
    onSweep: ({ sweep, cap, change, tol }) => {
      const converged = change !== undefined && tol !== undefined && change <= tol
      host.report({
        stage,
        done: converged ? cap : sweep,
        total: cap,
        converged,
        detail:
          change === undefined || tol === undefined
            ? `${unit} ${sweep} of ${cap}`
            : `${converged ? 'converged at ' : ''}${unit} ${sweep} of ${cap}; change ${exp(change)}, tolerance ${exp(tol)}`,
      })
    },
    stopped: host.stopped,
    yieldToPage: host.yieldToPage,
  }
}

/**
 * Steps a session to its end (or until it is stopped), one step at a time, reporting each to the hooks: a sampler or any
 * method that is not shown as a program. The shown programs write this call out in their own `infer`.
 *
 * @param run - The session, made in `live` mode.
 * @param cap - The steps it will take at most (the bar's total).
 * @param hooks - The page's hooks.
 * @returns How the run ended.
 */
export function stepSession<S, C extends Config, K extends string>(
  run: Session<S, C, K>,
  cap: number,
  hooks: EpHooks = {},
): Promise<AnytimeResult> {
  return runAnytimeAsync(run, {
    snapshot: { everySteps: 1 },
    onSnapshot: ({ t }) => hooks.onSweep?.({ sweep: t, cap }),
    shouldStop: () => hooks.stopped?.() === true,
    yieldToPage: hooks.yieldToPage ?? (() => Promise.resolve()),
  })
}

/**
 * A pacer for a loop of the page's own (Gibbs sweeps, seeds): call it after each round with the rounds done and the total; it
 * reports to the hooks, hands the thread back when `everyMs` has passed since it last did (so a cheap round costs nothing)
 * and says whether the run has been stopped.
 *
 * @param hooks - The page's hooks.
 * @param everyMs - How long to compute between hand-overs (default 40).
 * @returns The function to call after each round: true when the loop should stop.
 */
export function pacer(hooks: EpHooks, everyMs = 40): (done: number, cap: number) => Promise<boolean> {
  let last = performance.now()
  return async (done, cap) => {
    hooks.onSweep?.({ sweep: done, cap })
    if (hooks.stopped?.() === true) return true
    if (performance.now() - last >= everyMs) {
      await (hooks.yieldToPage ?? (() => Promise.resolve()))()
      last = performance.now()
    }
    return hooks.stopped?.() === true
  }
}
