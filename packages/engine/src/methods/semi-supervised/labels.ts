/**
 * Labels for semi-supervised learning: the `UNLABELLED` marker (-1, scikit-learn's convention), `hideLabels` for a
 * keyed, stratified choice of the few rows that keep their labels in demos, and the helpers that split semi-supervised
 * targets into labelled codes and classes.
 */

import { WhiteboxError } from '../../lib/base/index.ts'
import { type LearnData, targetVector } from '../../lib/learn/index.ts'
import { foldIn, type Key, permutation } from '../../lib/random/index.ts'
import { type TensorLike, toFlat } from '../../lib/tensor/index.ts'

/** The label of an unlabelled row (scikit-learn's convention for semi-supervised targets). */
export const UNLABELLED = -1

/** Options of {@link hideLabels}. */
export interface HideLabelsOptions {
  /** Labels kept per class (default 3); ignored when `fraction` is given. */
  readonly perClass?: number
  /** The fraction of each class's labels kept (at least one per class). */
  readonly fraction?: number
  readonly key: Key
}

/**
 * Hide most labels of a fully labelled target, as semi-supervised demos do: a keyed, stratified choice of rows keeps
 * its label and every other row becomes {@link UNLABELLED}.
 *
 * @param y - Class labels `[n]` (non-negative).
 * @param options - Labels kept per class (or a fraction) and the key.
 * @returns The targets with hidden labels set to -1.
 * @throws WhiteboxError `value` for negative labels or a fraction outside (0, 1].
 */
export function hideLabels(y: TensorLike, options: HideLabelsOptions): number[] {
  const values = Array.from(targetVector(y, 'hideLabels'))
  if (values.some((v) => v < 0)) throw new WhiteboxError('value', 'hideLabels: labels must be non-negative')
  const f = options.fraction
  if (f !== undefined && !(f > 0 && f <= 1)) throw new WhiteboxError('value', 'hideLabels: fraction must be in (0, 1]')
  const classes = [...new Set(values)].sort((a, b) => a - b)
  const out = values.map(() => UNLABELLED)
  for (const [c, label] of classes.entries()) {
    const rows = values.flatMap((v, i) => (v === label ? [i] : []))
    const keep = f === undefined ? Math.min(options.perClass ?? 3, rows.length) : Math.ceil(f * rows.length)
    const order = toFlat(permutation(foldIn(options.key, c), rows.length))
    for (let j = 0; j < Math.max(1, keep); j++) {
      const i = rows[order[j] ?? 0] ?? 0
      out[i] = label
    }
  }
  return out
}

/**
 * Learn data with most labels hidden (see {@link hideLabels}), keeping the rest of the dataset (feature and class
 * names) as it was.
 *
 * @param data - Labelled learn data.
 * @param options - As {@link hideLabels}.
 * @returns The data with semi-supervised targets.
 * @throws WhiteboxError `value` when the data has no targets.
 */
export function withHiddenLabels<D extends LearnData>(data: D, options: HideLabelsOptions): D {
  if (data.y === undefined) throw new WhiteboxError('value', 'withHiddenLabels: the data has no labels')
  return { ...data, y: hideLabels(data.y, options) }
}

/**
 * Split semi-supervised targets into the labelled rows' codes and the classes (unlabelled rows get code -1).
 *
 * @param y - Targets with -1 for unlabelled rows.
 * @param who - The caller (for errors).
 * @param n - The number of rows.
 * @returns The classes (ascending) and a code per row.
 * @throws WhiteboxError `value` when no row is labelled.
 */
export function semiSupervisedCodes(
  y: TensorLike,
  who: string,
  n: number,
): { readonly classes: number[]; readonly codes: number[] } {
  const values = Array.from(targetVector(y, who, n))
  const classes = [...new Set(values.filter((v) => v !== UNLABELLED))].sort((a, b) => a - b)
  if (classes.length === 0) throw new WhiteboxError('value', `${who}: no labelled rows`)
  const index = new Map(classes.map((c, i) => [c, i]))
  return { classes, codes: values.map((v) => index.get(v) ?? UNLABELLED) }
}
