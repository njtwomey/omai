/**
 * The exposition record of a point set that grows or moves over a run (`pointsSteps`), in two or three dimensions: the
 * iterates of a chaos game filling an attractor, the draws of a sampler, a cloud of particles. Each recorded step holds
 * the points it adds (when the record accumulates) or the whole set at that step (when it does not).
 */

import { MdmlError } from '../base/index.ts'

/** The points of one recorded step. */
export interface PointsStep {
  /** The tape step. */
  readonly step: number
  /** points[i]: the coordinates of point i (two or three numbers, as the record's `dims` says). */
  readonly points: readonly (readonly number[])[]
  /** A class per point (0, 1, ...), for colour. */
  readonly groups?: readonly number[]
  readonly caption?: string
}

/** Points at recorded steps, in two or three dimensions. */
export interface PointsSteps {
  readonly kind: 'pointsSteps'
  readonly dims: 2 | 3
  /** The coordinates' names (default `x`, `y`, `z`). */
  readonly features?: readonly string[]
  readonly steps: readonly PointsStep[]
  /**
   * Whether each step adds its points to those of the steps before it (default `true`: an attractor filling in), or
   * holds the whole set at that step (`false`: particles moving).
   */
  readonly accumulate?: boolean
}

/** The points shown at a step: their coordinates, groups and the caption of the step they come from. */
export interface PointsAt {
  readonly points: readonly (readonly number[])[]
  readonly groups?: readonly number[]
  readonly caption?: string
  /** How many of the points the latest step added (the rest are older). */
  readonly fresh: number
}

/**
 * Points at steps, with each point's dimension, the groups and the order of the steps checked.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws MdmlError `shape` when a point does not have `dims` coordinates or groups do not match the points; `value`
 *   when the steps are not in order.
 * @example
 * ```ts
 * import { pointsSteps } from 'mdml-engine/exposition'
 * const r = pointsSteps({ dims: 2, steps: [{ step: 0, points: [[0, 0]] }, { step: 1, points: [[1, 1], [2, 0]] }] })
 * r.steps.length // => 2
 * ```
 */
export function pointsSteps(record: Omit<PointsSteps, 'kind'>): PointsSteps {
  let previous = Number.NEGATIVE_INFINITY
  for (const [t, s] of record.steps.entries()) {
    if (s.step < previous) throw new MdmlError('value', `pointsSteps: steps[${t}] is out of order`)
    previous = s.step
    for (const [i, p] of s.points.entries())
      if (p.length !== record.dims)
        throw new MdmlError(
          'shape',
          `pointsSteps: steps[${t}].points[${i}] has ${p.length} entries, expected ${record.dims}`,
        )
    if (s.groups !== undefined && s.groups.length !== s.points.length)
      throw new MdmlError(
        'shape',
        `pointsSteps: steps[${t}].groups has ${s.groups.length} entries, expected ${s.points.length}`,
      )
  }
  return { kind: 'pointsSteps', ...record }
}

/**
 * The points to show at a step: those of every step up to it when the record accumulates, else those of the latest
 * step at or before it (the first when none is yet).
 *
 * @param record - The record.
 * @param step - The tape step.
 * @returns The points, their groups (when every step up to it has them) and the latest step's caption.
 * @example
 * ```ts
 * import { pointsAt, pointsSteps } from 'mdml-engine/exposition'
 * const r = pointsSteps({ dims: 2, steps: [{ step: 0, points: [[0, 0]] }, { step: 5, points: [[1, 1], [2, 0]] }] })
 * pointsAt(r, 3).points.length // => 1
 * pointsAt(r, 5).points.length // => 3
 * ```
 */
export function pointsAt(record: PointsSteps, step: number): PointsAt {
  const upTo = record.steps.filter((s) => s.step <= step)
  const used = upTo.length === 0 ? record.steps.slice(0, 1) : upTo
  const latest = used[used.length - 1]
  const from = record.accumulate === false ? used.slice(-1) : used
  const grouped = from.every((s) => s.groups !== undefined)
  return {
    points: from.flatMap((s) => s.points),
    ...(grouped && from.length > 0 ? { groups: from.flatMap((s) => s.groups ?? []) } : {}),
    ...(latest?.caption === undefined ? {} : { caption: latest.caption }),
    fresh: latest?.points.length ?? 0,
  }
}
