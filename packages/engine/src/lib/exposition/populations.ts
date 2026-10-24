import { rowsOf, sameLength } from './_check.ts'

/** One generation of an evolving population. */
export interface Generation {
  /** The fitness (or objective) of every individual. */
  readonly fitness: readonly number[]
  /** Every individual's position, for populations of points in the plane of the record's `surface`. */
  readonly points?: readonly (readonly [number, number])[]
  /** The search distribution's mean and covariance `[xx, xy, yy]` (evolution strategies, CMA-ES). */
  readonly mean?: readonly [number, number]
  readonly covariance?: readonly [number, number, number]
  /** The best individual of the generation: its fitness, text (an expression, a bit string), position or curve. */
  readonly best: {
    readonly fitness: number
    readonly label?: string
    readonly point?: readonly [number, number]
    /** Its predictions on the record's `data.grid` (a program fitted to data). */
    readonly curve?: readonly number[]
  }
}

/**
 * An evolving population, generation by generation (genetic algorithms, evolution strategies, CMA-ES, genetic
 * programming): every individual's fitness, and either its position over the objective's surface in the plane or the
 * best program's curve over the data it is fitted to.
 */
export interface PopulationSteps {
  readonly kind: 'populationSteps'
  readonly method: string
  /** Whether lower fitness is better (an objective minimised; default false: fitness maximised). */
  readonly lowerIsBetter?: boolean
  /** The objective over a grid of the plane: values[i][j] at (x[j], y[i]), with the axes' names. */
  readonly surface?: {
    readonly x: readonly number[]
    readonly y: readonly number[]
    readonly values: readonly (readonly number[])[]
    readonly names?: readonly [string, string]
    /** The optimum, when known. */
    readonly optimum?: readonly [number, number]
  }
  /** The data a program is fitted to, and the grid its curves are evaluated on. */
  readonly data?: { readonly x: readonly number[]; readonly y: readonly number[]; readonly grid: readonly number[] }
  readonly generations: readonly Generation[]
}

/**
 * Population steps, with their shapes checked.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws WhiteboxError `shape` when the surface does not match its axes, a generation's points do not match its
 *   fitness, or a curve does not match the data's grid.
 * @example
 * ```ts
 * import { populationSteps } from 'whitebox-engine/exposition'
 * const r = populationSteps({
 *   method: 'ga',
 *   generations: [{ fitness: [1, 3], best: { fitness: 3, label: '11' } }, { fitness: [3, 4], best: { fitness: 4 } }],
 * })
 * r.generations.length // => 2
 * ```
 */
export function populationSteps(record: Omit<PopulationSteps, 'kind'>): PopulationSteps {
  const s = record.surface
  if (s !== undefined) {
    sameLength('populationSteps', 'surface.values', s.values.length, s.y.length)
    rowsOf('populationSteps', 'surface.values', s.values, s.x.length)
  }
  const d = record.data
  if (d !== undefined) sameLength('populationSteps', 'data.y', d.y.length, d.x.length)
  for (const [t, g] of record.generations.entries()) {
    if (g.points !== undefined)
      sameLength('populationSteps', `generations[${t}].points`, g.points.length, g.fitness.length)
    if (g.best.curve !== undefined && d !== undefined)
      sameLength('populationSteps', `generations[${t}].best.curve`, g.best.curve.length, d.grid.length)
  }
  return { kind: 'populationSteps', ...record }
}
