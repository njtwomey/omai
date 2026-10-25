/**
 * Objectives on the plane for the evolutionary studies, private to this area: the core's test functions (Rastrigin,
 * Himmelblau, Rosenbrock, Ackley) with their box, a minimiser and their surface on a grid, so populations can be drawn
 * over contours.
 */

import { WhiteboxError } from '../../lib/base/index.ts'
import { ackley, himmelblau, rastrigin, rosenbrock, type TestFunction } from '../../lib/datasets/index.ts'
import { item, tensor, toFlat } from '../../lib/tensor/index.ts'

/** The objectives on the plane. */
export type PlaneFunctionName = 'rastrigin' | 'himmelblau' | 'rosenbrock' | 'ackley'

/** An objective on the plane (minimised), its box, a minimiser and its values on a grid. */
export interface PlaneFunction {
  readonly name: PlaneFunctionName
  readonly f: (x: readonly number[]) => number
  readonly lower: readonly [number, number]
  readonly upper: readonly [number, number]
  readonly optimum: readonly [number, number]
  readonly surface: {
    readonly x: readonly number[]
    readonly y: readonly number[]
    readonly values: readonly (readonly number[])[]
  }
}

const FUNCTIONS: Readonly<Record<PlaneFunctionName, () => TestFunction>> = {
  rastrigin: () => rastrigin(2),
  himmelblau,
  rosenbrock: () => rosenbrock(2),
  ackley: () => ackley(2),
}

/**
 * An objective on the plane with its surface sampled on a grid.
 *
 * @param name - The function.
 * @param resolution - Grid points per axis (default 61).
 * @returns The function.
 */
export function planeFunction(name: PlaneFunctionName, resolution = 61): PlaneFunction {
  const make = FUNCTIONS[name]
  if (make === undefined) throw new WhiteboxError('value', `planeFunction: unknown function '${name}'`)
  const t = make()
  // Ackley's usual box is mostly flat: look at its centre.
  const [bx = [-5, 5], by = [-5, 5]] = name === 'ackley' ? [[-5, 5] as const, [-5, 5] as const] : t.bounds
  const axis = (lo: number, hi: number): number[] =>
    Array.from({ length: resolution }, (_, i) => lo + ((hi - lo) * i) / (resolution - 1))
  const x = axis(bx[0], bx[1])
  const y = axis(by[0], by[1])
  const rows: number[][] = []
  for (const yi of y) rows.push(...x.map((xj) => [xj, yi]))
  const flat = Array.from(toFlat(t.f(tensor(rows))))
  const values = y.map((_, i) => flat.slice(i * x.length, (i + 1) * x.length))
  const m = t.minimizers[0] ?? [0, 0]
  return {
    name,
    f: (p) => item(t.f(tensor([p[0] ?? 0, p[1] ?? 0]))),
    lower: [bx[0], by[0]],
    upper: [bx[1], by[1]],
    optimum: [m[0] ?? 0, m[1] ?? 0],
    surface: { x, y, values },
  }
}
