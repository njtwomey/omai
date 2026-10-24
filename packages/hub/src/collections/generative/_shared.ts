// Shared settings of the generative hub entries (private to collections/generative): the two-dimensional datasets
// they learn, and reading numeric controls.
import type { HubEntry } from '../../entry/entry.ts'

/** The datasets of the two-dimensional generative entries, starting from two moons. */
export const GENERATIVE_2D: Pick<HubEntry, 'datasets' | 'dataset' | 'knobs'> = {
  datasets: ['density2d', 'clustering2d', 'classification2d'],
  dataset: 'moons',
}

/** A numeric control's value, or a fallback. */
export function num(v: unknown, fallback: number): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : fallback
}

/** A choice control's value among some options, or a fallback. */
export function pick<T extends string>(v: unknown, options: readonly T[], fallback: T): T {
  return typeof v === 'string' && (options as readonly string[]).includes(v) ? (v as T) : fallback
}
