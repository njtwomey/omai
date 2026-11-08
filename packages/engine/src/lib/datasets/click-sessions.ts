/**
 * Click sessions from the dynamic Bayesian network click model with its parameters as truth: a user scans a ranked
 * list from the top, examines each result in turn, clicks it if it is attractive, may be satisfied by a click and stop,
 * and otherwise continues to the next result with a fixed probability. The complement of `clickLogs` (position-based
 * and cascade models), whose logging policy it replaces by uniformly random lists.
 */

import { type DatasetGenerator, type InteractionData, interactions } from '../data/index.ts'
import { foldIn, permutation, uniform } from '../random/index.ts'
import { dict, int, real } from '../space/index.ts'
import { stamp } from './_util.ts'

/**
 * Click sessions from a dynamic Bayesian network (Chapelle and Zhang 2009). Each item has an attractiveness `a` and a
 * satisfaction `s`; a session shows `listLength` items in random order; the user examines the first, clicks an examined
 * item with probability `a`, a click satisfies with probability `s` (and ends the session), and an unsatisfied user
 * goes on to the next item with probability `continuation`. The data has one row per displayed item (session, rank,
 * click); the truth has the attractiveness and satisfaction of every item and the continuation probability.
 *
 * @example
 * ```ts
 * import { generate } from 'omai-engine/data'
 * import { dbnClickLogs } from 'omai-engine/datasets'
 * generate(dbnClickLogs, { knobs: { sessions: 10, listLength: 5 } }).data.positions?.shape // => [50]
 * ```
 */
export const dbnClickLogs: DatasetGenerator<InteractionData> = {
  name: 'dbnClickLogs',
  kind: 'interactions',
  description:
    'Click sessions from a dynamic Bayesian network click model with known attractiveness, satisfaction and continuation.',
  knobs: dict({
    sessions: int(1, 1_000_000, { label: 'sessions', description: 'The number of search sessions.', default: 2000 }),
    items: int(2, 10_000, { label: 'items', description: 'The number of distinct items.', default: 30 }),
    listLength: int(1, 100, { label: 'items shown per session', description: 'The length of each list.', default: 8 }),
    continuation: real(0, 1, {
      label: 'continuation',
      description: 'The chance an unsatisfied user goes on to the next result.',
      default: 0.85,
    }),
  }),
  generate: (k, knobs) => {
    const S = knobs['sessions'] as number
    const I = knobs['items'] as number
    const L = Math.min(knobs['listLength'] as number, I)
    const gamma = knobs['continuation'] as number
    const attractiveness = Float64Array.from(uniform(foldIn(k, 0), [I], { low: 0.1, high: 0.9 }).data)
    const satisfaction = Float64Array.from(uniform(foldIn(k, 1), [I], { low: 0.1, high: 0.9 }).data)
    const users: number[] = []
    const items: number[] = []
    const values: number[] = []
    const positions: number[] = []
    for (let s = 0; s < S; s++) {
      const order = Array.from(permutation(foldIn(foldIn(k, 2), s), I).data).slice(0, L)
      const u = uniform(foldIn(foldIn(k, 3), s), [3 * L]).data
      let examining: boolean = true
      for (let r = 0; r < L; r++) {
        const item = order[r] as number
        const clicked: boolean = examining && (u[3 * r] ?? 1) < (attractiveness[item] ?? 0)
        users.push(s)
        items.push(item)
        values.push(clicked ? 1 : 0)
        positions.push(r)
        if (examining) {
          const satisfied: boolean = clicked && (u[3 * r + 1] ?? 1) < (satisfaction[item] ?? 0)
          examining = !satisfied && (u[3 * r + 2] ?? 1) < gamma
        }
      }
    }
    return {
      data: interactions(
        { users, items, values, positions, sessions: users, nUsers: S, nItems: I },
        stamp('dbnClickLogs', dbnClickLogs.description, knobs),
      ),
      truth: { latents: { attractiveness, satisfaction }, extra: { continuation: gamma, model: 'dbn' } },
    }
  },
}
