// The default cross-validation folds of the search wrappers, as scikit-learn chooses them for `cv=<int>`: KFold for
// regression and StratifiedKFold for classification, both unshuffled. Not re-exported.
// TODO(learn): move to learn/model-selection/splitters.ts (kFold, stratifiedKFold) when the learn domain lands, and
// import from there.
import { WhiteboxError } from '../base/index.ts'
import type { Fold } from './estimators.ts'

function check(n: number, k: number, name: string): void {
  if (!Number.isInteger(k) || k < 2) throw new WhiteboxError('value', `${name}: need at least 2 folds, got ${k}`)
  if (k > n) throw new WhiteboxError('value', `${name}: ${k} folds for ${n} rows`)
}

/** sklearn KFold(k): contiguous test blocks, the first n % k of size floor(n / k) + 1. */
export function kFoldIndices(n: number, k: number): Fold[] {
  check(n, k, 'kFold')
  const out: Fold[] = []
  let start = 0
  for (let f = 0; f < k; f++) {
    const size = Math.floor(n / k) + (f < n % k ? 1 : 0)
    const test = Array.from({ length: size }, (_, i) => start + i)
    const train = Array.from({ length: n }, (_, i) => i).filter((i) => i < start || i >= start + size)
    out.push({ train, test })
    start += size
  }
  return out
}

/**
 * sklearn StratifiedKFold(k) without shuffling: classes in order of first appearance, the sorted class codes dealt to
 * the folds in turn, and each class's rows assigned to folds in row order by that allocation.
 */
export function stratifiedKFoldIndices(labels: readonly number[], k: number): Fold[] {
  const n = labels.length
  check(n, k, 'stratifiedKFold')
  const firstSeen = [...new Set([...labels].sort((a, b) => a - b))].map((c) => labels.indexOf(c))
  const order = [...firstSeen.keys()].sort((a, b) => (firstSeen[a] ?? 0) - (firstSeen[b] ?? 0))
  const sortedClasses = [...new Set([...labels].sort((a, b) => a - b))]
  const codeOf = new Map<number, number>()
  for (const [rank, i] of order.entries()) codeOf.set(sortedClasses[i] as number, rank)
  const encoded = labels.map((y) => codeOf.get(y) ?? 0)
  const nClasses = sortedClasses.length
  const sorted = [...encoded].sort((a, b) => a - b)
  const allocation = Array.from({ length: k }, (_, f) => {
    const counts = new Array<number>(nClasses).fill(0)
    for (let i = f; i < n; i += k) counts[sorted[i] ?? 0] = (counts[sorted[i] ?? 0] ?? 0) + 1
    return counts
  })
  const testFold = new Array<number>(n).fill(0)
  for (let c = 0; c < nClasses; c++) {
    const folds: number[] = []
    for (let f = 0; f < k; f++) for (let r = 0; r < (allocation[f]?.[c] ?? 0); r++) folds.push(f)
    let next = 0
    for (let i = 0; i < n; i++) if (encoded[i] === c) testFold[i] = folds[next++] ?? 0
  }
  return Array.from({ length: k }, (_, f) => ({
    train: testFold.flatMap((g, i) => (g === f ? [] : [i])),
    test: testFold.flatMap((g, i) => (g === f ? [i] : [])),
  }))
}
