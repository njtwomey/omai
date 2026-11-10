/**
 * Simulation-based calibration as a method: one replication per step (draw the parameters and the data, fit, rank the
 * truth), the ranks so far as the state, so a page can run it within a time budget, take a snapshot a second and show the
 * rank histograms and the ECDF difference fill in while it goes. It is the replications of {@link sbc} made steppable,
 * resumable and cancellable like the inference methods, with their budget and snapshot knobs; the result at any step is
 * the same tests of the ranks so far.
 */

import type { Json } from '../../base/index.ts'
import { type Key, key as makeKey } from '../../random/index.ts'
import { dict } from '../../space/index.ts'
import { type Config, defineMethod, type Method, scalar } from '../../trace/index.ts'
import { withAnytime } from '../inference/method.ts'
import { type SbcOptions, type SbcResult, sbcReplication, sbcSummary } from './sbc.ts'

/** The state of a calibration run: the ranks of each quantity so far, one per replication done. */
export interface SbcState {
  readonly ranks: Record<string, number[]>
  readonly done: number
}

/** What {@link sbcMethod} takes: the simulator and the fit of {@link sbc}, and how many replications it may take. */
export interface SbcMethodOptions<D>
  extends Pick<SbcOptions<D>, 'simulate' | 'fit' | 'numRanks' | 'bins' | 'alpha' | 'nullSimulations'> {
  /** The method's name (default `sbc`). */
  readonly name?: string
  /** The replications the run ends after (default 1000: the budget usually ends it first). */
  readonly replications?: number
  /** The key the replications are drawn from (default `key(0)`). */
  readonly key?: Key
}

/** The paths an SBC run logs. */
export type SbcPath = 'sbc/replications'

/** A calibration method, with a reading of its state as the tests of the ranks so far. */
export type SbcMethod = Method<SbcState, Config, SbcPath> & {
  /** The tests of the ranks of a state (a snapshot's, say), or `undefined` before the tenth replication. */
  result(state: SbcState): SbcResult | undefined
}

/**
 * Simulation-based calibration as a steppable method: a step is one replication, the tape carries the number done, the
 * state the ranks (so it serialises and resumes), and `result(state)` gives the histograms, tests and diagnosis of the ranks
 * so far from the tenth replication on. Run it with `runAnytimeAsync` and a time budget and read `result` at each snapshot.
 * The null distribution of the ECDF statistic is simulated with `nullSimulations` (default 400: a snapshot's p-value is a
 * guide, and a smaller number keeps it cheap).
 *
 * @param options - The simulator, the fit, `numRanks`, the key and the budget of replications.
 * @returns The method.
 * @concept posterior-validation
 * @index estimator Simulation-based calibration, one replication per step, with the tests of the ranks so far.
 * @example
 * ```ts
 * import { sbcMethod } from 'omai-engine/probprog/validation'
 * import { foldIn, normal } from 'omai-engine/random'
 * import { item } from 'omai-engine/tensor'
 * import { session } from 'omai-engine/trace'
 * const method = sbcMethod({
 *   numRanks: 19, replications: 40,
 *   simulate: (k) => { const m = item(normal(foldIn(k, 0), [])); return { data: m + item(normal(foldIn(k, 1), [])), truth: { m } } },
 *   fit: (y, k) => ({ m: Array.from(normal(k, [19], { loc: y / 2, scale: Math.SQRT1_2 }).data) }),
 * })
 * const s = session(method, { steps: 40, mode: 'eager' })
 * [s.t, method.result(s.state)?.passes] // => [40, true]
 * ```
 */
export function sbcMethod<D>(options: SbcMethodOptions<D>): SbcMethod {
  const total = options.replications ?? 1000
  const root = options.key ?? makeKey(0)
  const name = options.name ?? 'sbc'
  const result = (state: SbcState): SbcResult | undefined =>
    state.done < 10
      ? undefined
      : sbcSummary(state.ranks, {
          numRanks: options.numRanks,
          nullSimulations: options.nullSimulations ?? 400,
          ...(options.bins === undefined ? {} : { bins: options.bins }),
          ...(options.alpha === undefined ? {} : { alpha: options.alpha }),
        })
  const method = defineMethod<SbcState, Config, SbcPath>({
    name,
    config: withAnytime(dict({})),
    make: () => ({
      name,
      config: {},
      init: () => ({ ranks: {}, done: 0 }),
      step: (state, ctx) => {
        const ranks: Record<string, number[]> = { ...state.ranks }
        for (const [q, rank] of Object.entries(
          sbcReplication(
            { simulate: options.simulate, fit: options.fit, numRanks: options.numRanks, key: root },
            state.done,
          ),
        ))
          ranks[q] = [...(ranks[q] ?? []), rank]
        ctx.log('sbc/replications', state.done + 1)
        return { ranks, done: state.done + 1 }
      },
      done: (state) => (state.done >= total ? 'completed' : undefined),
      serialise: (state) => ({ ranks: state.ranks, done: state.done }) as unknown as Json,
      deserialise: (json) => json as unknown as SbcState,
    }),
    contract: { 'sbc/replications': scalar({ description: 'the replications done' }) },
  })
  return { ...method, result }
}
