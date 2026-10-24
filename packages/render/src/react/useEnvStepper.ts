import { useCallback, useMemo, useState } from 'react'
import { foldIn, key as makeKey } from 'whitebox-engine/random'
import type { Env } from 'whitebox-engine/rl'

/** What {@link useEnvStepper} returns: the episode so far and the actions that drive it. */
export interface EnvStepper<S, O, A> {
  readonly state: S
  readonly observation: O
  /** Steps taken since the reset. */
  readonly t: number
  readonly lastReward: number
  readonly return: number
  readonly done: boolean
  readonly step: (action: A) => void
  readonly reset: () => void
}

/**
 * Step an environment by hand: reset from a seed, then `step(action)` advances it with keys derived from the seed and
 * the step, so the same seed and actions always give the same episode. Stepping after the end does nothing until a
 * reset.
 *
 * @param env - The environment.
 * @param seed - The seed (default 0); a new seed resets.
 * @returns The state, its observation and running totals, and `step` and `reset`.
 */
export function useEnvStepper<S, O, A>(env: Env<S, O, A>, seed = 0): EnvStepper<S, O, A> {
  const root = useMemo(() => makeKey(seed), [seed])
  const start = useCallback(() => {
    const r = env.reset(foldIn(root, 0))
    return { state: r.state, observation: r.observation, t: 0, lastReward: 0, total: 0, done: false }
  }, [env, root])
  const [run, setRun] = useState(start)
  const [owner, setOwner] = useState(() => start)
  // A new environment or seed starts a new episode.
  if (owner !== start) {
    setOwner(() => start)
    setRun(start())
  }
  return {
    state: run.state,
    observation: run.observation,
    t: run.t,
    lastReward: run.lastReward,
    return: run.total,
    done: run.done,
    step: (action) =>
      setRun((r) => {
        if (r.done) return r
        const s = env.step(r.state, action, foldIn(foldIn(root, 2), r.t))
        return {
          state: s.state,
          observation: s.observation,
          t: r.t + 1,
          lastReward: s.reward,
          total: r.total + s.reward,
          done: s.terminated || s.truncated,
        }
      }),
    reset: () => setRun(start()),
  }
}
