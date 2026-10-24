// Constraint-satisfaction hub entries (private to this collection): the solver's settings as controls.
import type { CspPlayOptions } from 'whitebox-engine/methods/gofai'
import { choice, type Space, type SpaceValue } from 'whitebox-engine/space'

/** The backtracking controls: variable ordering, value ordering and inference. */
export const backtrackingControls: Readonly<Record<string, Space>> = {
  variableOrder: choice(['mrv', 'mrv-degree', 'degree', 'static'], { default: 'mrv', label: 'variable order' }),
  valueOrder: choice(['static', 'lcv'], { default: 'static', label: 'value order' }),
  inference: choice(['forward-checking', 'mac', 'none'], { default: 'forward-checking', label: 'inference' }),
}

/**
 * The solver options of an entry's parameters.
 *
 * @param params - The entry's parameter values.
 * @returns The options for the played solvers.
 */
export function solverOptions(params: Readonly<Record<string, SpaceValue>>): CspPlayOptions {
  const solver = params['solver'] === 'min-conflicts' ? 'min-conflicts' : 'backtracking'
  return {
    solver,
    ...(params['variableOrder'] === undefined ? {} : { variableOrder: params['variableOrder'] as 'mrv' }),
    ...(params['valueOrder'] === undefined ? {} : { valueOrder: params['valueOrder'] as 'lcv' }),
    ...(params['inference'] === undefined ? {} : { inference: params['inference'] as 'mac' }),
  }
}
