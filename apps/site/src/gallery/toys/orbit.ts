/**
 * A small toy method for the export and narrative recipes: a point walking a wobbly circle, whose laps are an event of
 * its contract. It logs the angle, the radius, the lap count (when a lap ends) and the position, and has one derived path,
 * the distance from the point over the plane around it.
 */

import { dict, int, real } from 'mdml-engine/space'
import {
  type Config,
  defineMethod,
  derived,
  event,
  Log,
  type Method,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from 'mdml-engine/trace'

/** Where the walk stands. */
export interface Walk {
  readonly k: number
  readonly x: number
  readonly y: number
}

/** Steps of a run. */
export const ORBIT_STEPS = 200

const GRID = Array.from({ length: 25 }, (_, i) => -1.5 + 0.125 * i)

/** The walk. */
export const orbit: Method<Walk, Config> = defineMethod<Walk, Config>({
  name: 'orbit',
  config: dict({
    laps: int(1, 8, { default: 4, label: 'laps' }),
    wobble: real(0, 0.5, { default: 0.2, label: 'wobble of the radius' }),
  }),
  make: (c) => {
    const laps = c['laps'] as number
    const wobble = c['wobble'] as number
    const perLap = ORBIT_STEPS / laps
    return {
      name: 'orbit',
      config: { laps, wobble },
      init: () => ({ k: 0, x: 1, y: 0 }),
      step: (s, ctx) => {
        const k = s.k + 1
        const a = (2 * Math.PI * k) / perLap
        const r = 1 + wobble * Math.sin(5 * a)
        ctx.log('angle', a)
        ctx.log('radius', r)
        ctx.log('position', [r * Math.cos(a), r * Math.sin(a)])
        if (k % perLap === 0) {
          ctx.log('laps', k / perLap)
          ctx.emit('lap/end', { lap: k / perLap })
        }
        return { k, x: r * Math.cos(a), y: r * Math.sin(a) }
      },
      done: (s) => (s.k >= ORBIT_STEPS ? 'completed' : undefined),
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as Walk,
    }
  },
  contract: {
    angle: scalar({ description: 'the angle walked' }),
    radius: scalar({ description: 'the distance from the origin' }),
    laps: scalar({ description: 'the lap just finished' }),
    position: tensor({ role: 'state', shape: [2], description: 'where the point is' }),
    'lap/end': event({ context: { lap: 'the lap just finished' }, description: 'a lap ends' }),
    'plane/distance': derived<Walk>({
      description: 'the distance from the point, over the plane',
      compute: (s) => ({
        kind: 'decisionSurface',
        features: ['x', 'y'],
        x: GRID,
        y: GRID,
        values: GRID.map((y) => GRID.map((x) => Math.hypot(x - s.x, y - s.y))),
        field: { name: 'distance' },
      }),
    }),
  },
  defaults: { angle: Log(), radius: Log(), laps: Log(), position: Log({ detail: 'full' }) },
})
