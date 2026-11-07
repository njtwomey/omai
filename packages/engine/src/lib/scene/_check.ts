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

/** Fail unless every vertex is `dims` finite numbers (2-D scene vertices, 3-D world points). */
export function vertices(who: string, points: readonly (readonly number[])[], dims: 2 | 3): void {
  for (const [i, p] of points.entries())
    if (!(p.length === dims && p.every(Number.isFinite)))
      throw new MdmlError('value', `${who}: points[${i}] is [${p}], expected ${dims} finite numbers`)
}

/** Fail unless a group's transform is finite and its scale not zero (a zero scale has no inverse). */
export function transform(
  who: string,
  t: { readonly translate?: readonly [number, number]; readonly rotate?: number; readonly scale?: number },
): void {
  if (t.translate !== undefined) vertices(who, [t.translate], 2)
  if (t.rotate !== undefined) finite(who, { rotate: t.rotate })
  if (t.scale !== undefined && !(Number.isFinite(t.scale) && t.scale !== 0))
    throw new MdmlError('value', `${who}: scale is ${t.scale}, expected a finite number other than 0`)
}
