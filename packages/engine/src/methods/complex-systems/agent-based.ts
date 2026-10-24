// Agent-based models played round by round: Schelling's segregation model.
import { WhiteboxError } from '../../lib/base/index.ts'
import { type BoardSteps, boardSteps, type Explained } from '../../lib/exposition/index.ts'
import { type Key, key as makeKey, permutation, split, uniform } from '../../lib/random/index.ts'

/** Options of {@link schelling}. */
export interface SchellingOptions {
  /** The side of the square town (default 24). */
  readonly size?: number
  /** The share of empty houses (default 0.1). */
  readonly vacancy?: number
  /** The share of like neighbours an agent wants (default 0.4). */
  readonly threshold?: number
  /** Rounds (default 30); each round every unhappy agent moves to a random empty house. */
  readonly rounds?: number
  readonly key?: Key
}

/** What a Schelling run ended with. */
export interface SchellingResult {
  /** 0 empty, 1 and 2 the two groups, per house. */
  readonly town: readonly number[]
  /** The share of unhappy agents and the mean share of like neighbours, per round. */
  readonly unhappy: readonly number[]
  readonly similarity: readonly number[]
}

/**
 * Schelling's segregation model: two groups on a grid, each agent content when at least a threshold share of its
 * occupied neighbours (8-neighbourhood) are like it; each round every unhappy agent moves to a random empty house.
 * Mild preferences produce strong segregation. Recorded round by round, unhappy agents marked.
 *
 * @param options - Town size, vacancy, threshold, rounds and key.
 * @returns The final town and the per-round shares, with a `boardSteps` record.
 * @throws WhiteboxError `value` for a vacancy or threshold outside [0, 1].
 */
export function schelling(options: SchellingOptions = {}): Explained<SchellingResult> {
  const size = options.size ?? 24
  const vacancy = options.vacancy ?? 0.1
  const threshold = options.threshold ?? 0.4
  if (!(vacancy > 0 && vacancy < 1 && threshold >= 0 && threshold <= 1))
    throw new WhiteboxError('value', 'schelling: vacancy must be in (0, 1) and threshold in [0, 1]')
  const n = size * size
  const [k0, ...rest] = split(options.key ?? makeKey(0), 2 + (options.rounds ?? 30)) as Key[]
  const u = uniform(k0 as Key, [n]).data
  let town = Array.from(u, (x) => (x < vacancy ? 0 : x < vacancy + (1 - vacancy) / 2 ? 1 : 2))
  const like = (t: readonly number[], i: number): { same: number; others: number } => {
    const r = Math.floor(i / size)
    const c = i % size
    let same = 0
    let others = 0
    for (let dr = -1; dr <= 1; dr++)
      for (let dc = -1; dc <= 1; dc++) {
        if ((dr === 0 && dc === 0) || r + dr < 0 || r + dr >= size || c + dc < 0 || c + dc >= size) continue
        const v = t[(r + dr) * size + c + dc] ?? 0
        if (v === 0) continue
        if (v === t[i]) same += 1
        else others += 1
      }
    return { same, others }
  }
  const unhappyOf = (t: readonly number[]): boolean[] =>
    t.map((v, i) => {
      if (v === 0) return false
      const { same, others } = like(t, i)
      return same + others > 0 && same / (same + others) < threshold
    })
  const similarityOf = (t: readonly number[]): number => {
    let s = 0
    let m = 0
    for (let i = 0; i < n; i++) {
      if (t[i] === 0) continue
      const { same, others } = like(t, i)
      if (same + others === 0) continue
      s += same / (same + others)
      m += 1
    }
    return m === 0 ? 0 : s / m
  }
  const towns: number[][] = [town]
  const unhappyFlags: boolean[][] = [unhappyOf(town)]
  for (const k of rest.slice(0, options.rounds ?? 30)) {
    const flags = unhappyFlags[unhappyFlags.length - 1] as boolean[]
    const movers = flags.flatMap((f, i) => (f ? [i] : []))
    if (movers.length === 0) break
    const next = [...town]
    const order = Array.from(permutation(k, movers.length).data, (j) => movers[j] as number)
    const empty = next.flatMap((v, i) => (v === 0 ? [i] : []))
    const picks = uniform(split(k, 2)[1] as Key, [order.length]).data
    for (const [j, i] of order.entries()) {
      const at = Math.floor((picks[j] ?? 0) * empty.length)
      const to = empty[at]
      if (to === undefined) break
      next[to] = next[i] ?? 0
      next[i] = 0
      empty[at] = i
    }
    town = next
    towns.push(town)
    unhappyFlags.push(unhappyOf(town))
  }
  const unhappy = unhappyFlags.map(
    (f, t) => f.filter(Boolean).length / Math.max(1, (towns[t] ?? []).filter((v) => v > 0).length),
  )
  const similarity = towns.map(similarityOf)
  const record: BoardSteps = boardSteps({
    method: 'schelling',
    layout: { type: 'grid', rows: size, columns: size },
    categories: ['empty', 'blue', 'red'],
    steps: towns.map((t, r) => ({
      cells: t,
      labels: (unhappyFlags[r] ?? []).map((f) => (f ? '×' : '')),
      caption: `round ${r}: ${Math.round(100 * (unhappy[r] ?? 0))}% unhappy, ${Math.round(100 * (similarity[r] ?? 0))}% like neighbours`,
    })),
    metrics: [
      { name: 'like neighbours', values: similarity },
      { name: 'unhappy', values: unhappy },
    ],
    stepName: 'round',
  })
  return { town, unhappy, similarity, exposition: [record] }
}
