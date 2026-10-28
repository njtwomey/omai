/**
 * Sessions with extra verbs: `extendSession` delegates a session's verbs one by one (`t`, `state` and `status` are
 * getters, so no spread) and adds what a base offers beside them (the fitted value at a step, the agent at a step),
 * keeping the extras on the sessions forked from it.
 */

import type { Config, Session } from '../../trace/index.ts'

/** A session with extra verbs, which its forks keep. */
export type SessionWith<S, X> = Omit<Session<S, Config>, 'fork'> &
  X & {
    fork(t: number, options?: { readonly seed?: number; readonly config?: Partial<Config> }): SessionWith<S, X>
  }

/** The session `s` with the verbs `extra` builds from it, on its forks too. */
export function extendSession<S, X extends object>(
  s: Session<S, Config>,
  extra: (s: Session<S, Config>) => X,
): SessionWith<S, X> {
  return {
    method: s.method,
    config: s.config,
    seed: s.seed,
    algorithm: s.algorithm,
    contract: s.contract,
    get t() {
      return s.t
    },
    get state() {
      return s.state
    },
    get status() {
      return s.status
    },
    advance: s.advance,
    stateAt: s.stateAt,
    fork: (t, options) => extendSession(s.fork(t, options), extra),
    probe: s.probe,
    project: s.project,
    run: s.run,
    stepsOf: s.stepsOf,
    ...extra(s),
  }
}
