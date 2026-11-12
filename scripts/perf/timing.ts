/**
 * The clock-reading part of the performance tools (`scripts/ep-perf.ts`, `scripts/perf/*.ts`): a wall-clock `now`, the calibration
 * workload and `bench`, which times a function as microseconds per call with a warm-up and batching. Scripts outside `packages/` may
 * read the clock (`scripts/banned.ts` covers published source only); the pure statistics are in `scripts/ep-perf-stats.ts`.
 */

/** The wall clock in milliseconds. */
export const now = (): number => performance.now()

/** Keeps the results of timed calls alive, so the engine cannot drop the work. */
export let sink = 0

/** A fixed pure-arithmetic workload (exponentials, logarithms, divisions in a scalar loop): the figure that tells machines and loads apart. */
function calibrationWork(): number {
  let s = 0
  for (let i = 1; i <= 2_000_000; i++) s += (Math.exp(-i * 1e-7) * Math.log(1 + i)) / (1 + (i % 7))
  return s
}

/**
 * The calibration workload timed `n` times (after one warm-up).
 *
 * @param n - The number of timings.
 * @returns Milliseconds of each.
 */
export function calibrate(n = 5): number[] {
  sink += calibrationWork()
  const ts: number[] = []
  for (let i = 0; i < n; i++) {
    const t = now()
    sink += calibrationWork()
    ts.push(now() - t)
  }
  return ts
}

/**
 * Microseconds per call of `fn`: warmed up, then `samples` timings, each of as many calls as fill `targetMs`.
 *
 * @param fn - The call.
 * @param samples - How many timings (the median is the figure).
 * @param targetMs - The least time one timing runs (calls are batched up to it, so a microsecond call is measured above the timer's floor).
 * @returns The microseconds per call of each timing.
 */
export function bench(fn: () => unknown, samples: number, targetMs: number): number[] {
  for (let i = 0; i < 3; i++) sink += Number(fn() !== undefined)
  let reps = 1
  for (;;) {
    const t = now()
    for (let i = 0; i < reps; i++) sink += Number(fn() !== undefined)
    const dt = now() - t
    if (dt >= targetMs || reps >= 1 << 22) break
    reps = Math.max(reps * 2, Math.ceil((reps * targetMs * 1.1) / Math.max(dt, 0.001)))
  }
  const us: number[] = []
  for (let s = 0; s < samples; s++) {
    const t = now()
    for (let i = 0; i < reps; i++) sink += Number(fn() !== undefined)
    us.push(((now() - t) * 1000) / reps)
  }
  return us
}

/** Keeps a number of a timed call alive (the importer of `sink` cannot assign to it). */
export function keep(x: number): void {
  sink += x
}
