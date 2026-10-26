// Ask-and-tell for the optimisers of `optim`, which call their objective themselves: to ask for the next point, replay
// the optimiser from its start, answering its first calls with the values the study has told it, and stop it at the
// first call it makes beyond them; that call's argument is the next point. The optimisers are deterministic given
// their keys, so the replay retraces the same calls, and the sampler stays a pure function of the study (resuming a
// study needs no sampler state). When the optimiser stops (it converged), it restarts from a new key. Not re-exported.
import { foldIn, type Key } from '../random/index.ts'
import { asTensor, toFlat, type Value } from '../tensor/index.ts'
import { type Algorithm, nullTracer, startRun, stepContext } from '../trace/index.ts'

/** A function of a flat vector, as the optimisers take it (a parameter tree; here always a vector). */
type Probe = (x: unknown) => Value

class Asked {
  readonly x: number[]
  constructor(x: number[]) {
    this.x = x
  }
}

/**
 * The next point an optimiser asks for after the told values, or `undefined` when it cannot ask (it stops before
 * using them, for instance with a degenerate start).
 *
 * @param make - The optimiser for an objective, a restart index and the root key.
 * @param told - The values of the points it asked for so far, in order (finite; failures as a penalty).
 * @param key - The root key of the optimiser runs.
 * @param maxRestarts - Restarts allowed before giving up.
 */
export function replayAsk<S>(
  make: (f: Probe, restart: number) => Algorithm<S>,
  told: readonly number[],
  key: Key,
  maxRestarts = 1000,
): number[] | undefined {
  let used = 0
  const f: Probe = (x) => {
    const v = told[used]
    if (v !== undefined) {
      used += 1
      return asTensor(v)
    }
    throw new Asked(toFlat(x as Value))
  }
  const run = startRun(nullTracer, { name: 'replay', seed: null, config: {} })
  try {
    for (let restart = 0; restart <= maxRestarts; restart++) {
      const before = used
      const algorithm = make(f, restart)
      const root = foldIn(key, restart)
      const steps = foldIn(root, 1)
      let state = algorithm.init(foldIn(root, 0))
      let stalled = 0
      for (let t = 0; algorithm.done?.(state, t) === undefined; t++) {
        const at = used
        state = algorithm.step(state, stepContext(t, foldIn(steps, t), run))
        stalled = used === at ? stalled + 1 : 0
        // Steps that evaluate nothing cannot ask for anything: give up rather than loop.
        if (stalled > 1000) return undefined
      }
      // A restart that used no told value would stop the same way again.
      if (used === before) return undefined
    }
  } catch (error) {
    if (error instanceof Asked) return error.x
    throw error
  }
  return undefined
}
