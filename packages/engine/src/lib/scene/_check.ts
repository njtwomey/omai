// Argument checks shared by the scene builders (private to the domain): a drawing with a NaN coordinate or a negative
// size is a bug where it is made, not a blank in SVG and a null in JSON.
import { MdmlError } from '../base/index.ts'

/** Fail unless every named number is finite. */
export function finite(who: string, fields: Readonly<Record<string, number>>): void {
  for (const [name, v] of Object.entries(fields))
    if (!Number.isFinite(v)) throw new MdmlError('value', `${who}: ${name} is ${v}, expected a finite number`)
}

/** Fail unless every named number is finite and not negative (a size). */
export function size(who: string, fields: Readonly<Record<string, number>>): void {
  for (const [name, v] of Object.entries(fields))
    if (!(Number.isFinite(v) && v >= 0)) throw new MdmlError('value', `${who}: ${name} is ${v}, expected a size >= 0`)
}
