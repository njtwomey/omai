// Argument checks shared by the hazard and segment-model files of this folder. Not re-exported.
import { MdmlError } from '../../base/index.ts'

/** Throw `value` unless a number is positive and finite. */
export function positive(who: string, name: string, v: number): void {
  if (!(v > 0 && Number.isFinite(v))) throw new MdmlError('value', `${who}: ${name} must be positive`)
}
