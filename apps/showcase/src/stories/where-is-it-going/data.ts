/**
 * The data of 'Where is it going?': a made-up object that moves along a line, so the page can say where it really was. It
 * starts near zero with some velocity; at every step its velocity is pushed a little at random and its position moves by
 * the velocity. A sensor reads the position at each step with noise. The readings are the same every time, so the page shows
 * them before play; the true positions are shown after the run, and the steps after the last reading are drawn from the
 * run's seed.
 */

import { scan } from 'omai-engine/nn'
import { foldIn, key, normal, split } from 'omai-engine/random'
import { add, concat, expandDims, mul, sliceAxis, stack, take, tensor, toFlat, type Value } from 'omai-engine/tensor'

/** The time between two readings. */
export const DT = 0.5

/** The start of the state: the mean and the standard deviation of the position and the velocity at the first step. */
export const START: { readonly mean: readonly [number, number]; readonly sd: readonly [number, number] } = {
  mean: [0, 1],
  sd: [2, 1],
}

/** The standard deviation of each reading's noise. */
export const READING_SD = 1

/** How strongly the velocity is pushed around at each step (the intensity of a random acceleration). */
export const PUSH = 0.5

/** How many readings there are, and how many further steps are predicted. */
export const READINGS = 40
export const AHEAD = 10

/** The covariance of the random push in one step: it moves the position and the velocity together. */
export const stepCovariance = (push: number = PUSH): number[][] => [
  [(push * DT ** 3) / 3, (push * DT ** 2) / 2],
  [(push * DT ** 2) / 2, push * DT],
]

/** The made-up track: the true position and velocity at each step, and the readings of the position. */
export interface Track {
  readonly position: readonly number[]
  readonly velocity: readonly number[]
  readonly readings: readonly number[]
}

/** The seed the first readings are made from: the same track every time, so the page can show the readings before anything runs. */
export const TRACK_SEED = 3

/**
 * The track: `steps` steps that are the same every time, with the readings of their positions, and `ahead` more steps whose pushes
 * are drawn from the run's seed, so the page can show where the object really went after the last reading.
 *
 * @param seed - Decides the pushes after the last reading.
 * @param steps - How many readings.
 * @param ahead - How many further steps to simulate.
 * @returns The track; the readings are as long as `steps`, the positions as long as `steps + ahead`.
 */
export function simulate(seed: number, steps: number = READINGS, ahead: number = AHEAD): Track {
  const [kx, kw, ke] = split(key(TRACK_SEED), 3) as [
    ReturnType<typeof key>,
    ReturnType<typeof key>,
    ReturnType<typeof key>,
  ]
  const total = steps + ahead
  const start = normal(kx, [2])
  // the pushes up to the last reading are fixed, and the pushes after it come from the seed
  const shocks = concat([normal(kw, [steps, 2]), normal(foldIn(key(seed), 0), [ahead, 2])])
  // the push is correlated between the position and the velocity: its Cholesky factor mixes two standard normals
  const q = stepCovariance()
  const l00 = Math.sqrt(q[0]?.[0] as number)
  const l10 = (q[1]?.[0] as number) / l00
  const l11 = Math.sqrt((q[1]?.[1] as number) - l10 ** 2)
  const first = add(tensor([...START.mean]), mul(tensor([...START.sd]), start))
  // the position moves by the velocity and the push; the velocity by the push
  const path = scan(
    ([p, v], shock) => {
      const a = take(shock, 0)
      const b = take(shock, 1)
      const position = add(add(p, mul(v, DT)), mul(a, l00))
      const velocity = add(add(v, mul(a, l10)), mul(b, l11))
      return [[position, velocity] as [Value, Value], stack([position, velocity])]
    },
    [take(first, 0), take(first, 1)] as [Value, Value],
    sliceAxis(shocks, 0, 1, total),
  )
  const states = concat([expandDims(first, 0), path.ys], 0)
  const position = take(states, 0, 1)
  const noisy = add(sliceAxis(position, 0, 0, steps), mul(normal(ke, [steps]), READING_SD))
  return { position: toFlat(position), velocity: toFlat(take(states, 1, 1)), readings: toFlat(noisy) }
}
