// Internal: when the clock of a live run plays and when it waits for the producer (the preroll).

/**
 * What the clock of a live, unfinished run does at a tick. Waiting (`buffering`), it plays again once `preroll` steps
 * stand ahead of it, or once it has waited `maxMs` and there is at least one step to show (a method too slow to fill
 * the preroll plays on what it has); playing, it starts waiting when it has caught up with the head.
 *
 * @param o - Whether it is waiting, the steps ahead of the cursor, the preroll in steps, how long it has waited and the
 *   longest wait.
 * @returns `play` to move the cursor, `wait` to stay where it is, `starved` to stand at the head and start waiting.
 */
export function prerollDecision(o: {
  readonly buffering: boolean
  readonly ahead: number
  readonly preroll: number
  readonly waitedMs: number
  readonly maxMs: number
}): 'play' | 'wait' | 'starved' {
  if (o.buffering) return o.ahead >= o.preroll || (o.ahead >= 1 && o.waitedMs >= o.maxMs) ? 'play' : 'wait'
  return o.ahead <= 0 ? 'starved' : 'play'
}
