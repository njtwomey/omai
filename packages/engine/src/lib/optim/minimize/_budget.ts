// The evaluation budget of the derivative-free methods (not exported from the barrel).

// Thrown by a counted objective when the evaluation budget is spent (scipy's _MaxFuncCallError).
export const BUDGET: unique symbol = Symbol('evaluation budget spent')

/** A counted objective: throws BUDGET once `max` evaluations (including `used` earlier ones) have been made. */
export function counted(value: (x: ArrayLike<number>) => number, used: number, max: number) {
  let calls = 0
  return {
    f: (x: ArrayLike<number>): number => {
      if (used + calls >= max) throw BUDGET
      calls += 1
      return value(x)
    },
    calls: (): number => calls,
  }
}

export const isBudget = (e: unknown): boolean => e === BUDGET
