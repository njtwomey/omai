// The objective control shared by the entries on the plane (private to this collection).
import { type ChoiceSpace, choice } from 'mdml-engine/space'

/**
 * The objective to minimise.
 *
 * @param start - The default.
 * @returns The control.
 */
export function objectiveControl(start: 'rastrigin' | 'himmelblau' | 'rosenbrock' | 'ackley'): ChoiceSpace {
  return choice(['rastrigin', 'himmelblau', 'rosenbrock', 'ackley'], { default: start, label: 'objective' })
}
