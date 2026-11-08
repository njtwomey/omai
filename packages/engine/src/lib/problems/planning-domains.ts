/**
 * Planning domains for the STRIPS planners of `omai-engine/search`: the blocks world with a gripper (pick up, put
 * down, stack, unstack), the Sussman anomaly, random blocks problems, and `towersOf` for reading a state back as towers
 * to draw.
 */

import { OmaiError } from '../base/index.ts'
import { type Key, permutation, split, uniform } from '../random/index.ts'
import { type PlanState, type StripsAction, type StripsProblem, strips } from '../search/index.ts'

/** A blocks-world configuration: towers of blocks, each listed bottom to top. */
export type Towers = readonly (readonly string[])[]

/**
 * The blocks world with a gripper (the four-operator STRIPS version): `pickup` and `putdown` take a block from and to
 * the table, `unstack` and `stack` take it from and onto another block; facts are `on(x,y)`, `ontable(x)`, `clear(x)`,
 * `holding(x)` and `handempty`.
 *
 * @param initial - The towers at the start.
 * @param goal - The towers wanted (every block in them is placed as shown).
 * @returns The STRIPS problem.
 * @throws OmaiError `value` when the two configurations hold different blocks or a block appears twice.
 */
export function blocksWorld(initial: Towers, goal: Towers): StripsProblem {
  const blocks = initial.flat()
  if (new Set(blocks).size !== blocks.length) throw new OmaiError('value', 'blocksWorld: a block appears twice')
  if (!goal.flat().every((b) => blocks.includes(b)))
    throw new OmaiError('value', 'blocksWorld: the goal names a block not in the initial towers')
  const facts = (towers: Towers, complete: boolean): string[] => {
    const out: string[] = complete ? ['handempty'] : []
    for (const t of towers)
      t.forEach((b, i) => {
        out.push(i === 0 ? `ontable(${b})` : `on(${b},${t[i - 1]})`)
        if (complete && i === t.length - 1) out.push(`clear(${b})`)
      })
    return out
  }
  const actions: StripsAction[] = []
  for (const x of blocks) {
    actions.push(
      {
        name: `pickup(${x})`,
        pre: [`clear(${x})`, `ontable(${x})`, 'handempty'],
        add: [`holding(${x})`],
        del: [`clear(${x})`, `ontable(${x})`, 'handempty'],
      },
      {
        name: `putdown(${x})`,
        pre: [`holding(${x})`],
        add: [`clear(${x})`, `ontable(${x})`, 'handempty'],
        del: [`holding(${x})`],
      },
    )
    for (const y of blocks) {
      if (x === y) continue
      actions.push(
        {
          name: `unstack(${x},${y})`,
          pre: [`on(${x},${y})`, `clear(${x})`, 'handempty'],
          add: [`holding(${x})`, `clear(${y})`],
          del: [`on(${x},${y})`, `clear(${x})`, 'handempty'],
        },
        {
          name: `stack(${x},${y})`,
          pre: [`holding(${x})`, `clear(${y})`],
          add: [`on(${x},${y})`, `clear(${x})`, 'handempty'],
          del: [`holding(${x})`, `clear(${y})`],
        },
      )
    }
  }
  return strips(facts(initial, true), facts(goal, false), actions)
}

/** The Sussman anomaly: C on A, B alone; the goal is A on B on C. */
export const SUSSMAN: { readonly initial: Towers; readonly goal: Towers } = {
  initial: [['A', 'C'], ['B']],
  goal: [['C', 'B', 'A']],
}

/** The towers of a blocks-world state, and the block held: towers in order of their bottom block. */
export interface BlocksPicture {
  readonly towers: Towers
  readonly holding: string | null
}

/**
 * Read the towers back from a blocks-world state.
 *
 * @param state - The facts that hold.
 * @returns The towers (by bottom block, alphabetically) and the block in the gripper.
 */
export function towersOf(state: PlanState): BlocksPicture {
  const above = new Map<string, string>()
  const bottoms: string[] = []
  let holding: string | null = null
  for (const f of state) {
    const m = /^(\w+)\(([^)]*)\)$/.exec(f)
    if (m === null) continue
    const [, p, args = ''] = m
    const [a = '', b = ''] = args.split(',')
    if (p === 'on') above.set(b, a)
    else if (p === 'ontable') bottoms.push(a)
    else if (p === 'holding') holding = a
  }
  const towers = bottoms.sort().map((b) => {
    const t = [b]
    for (let x = above.get(b); x !== undefined; x = above.get(x)) t.push(x)
    return t
  })
  return { towers, holding }
}

/**
 * A random blocks-world problem: `n` blocks dealt into random towers, twice (start and goal).
 *
 * @param key - The key.
 * @param n - The number of blocks (1 to 8, default 5).
 * @returns The two configurations.
 */
export function randomBlocks(key: Key, n = 5): { readonly initial: Towers; readonly goal: Towers } {
  if (!(Number.isInteger(n) && n >= 1 && n <= 8)) throw new OmaiError('value', `randomBlocks: n = ${n} must be 1 to 8`)
  const names = Array.from({ length: n }, (_, i) => String.fromCharCode(65 + i))
  const deal = (k: Key): string[][] => {
    const [a, b] = split(k, 2) as [Key, Key]
    const order = Array.from(permutation(a, n).data, (i) => names[i] ?? '')
    const cuts = Array.from(uniform(b, [n]).data)
    const towers: string[][] = [[]]
    order.forEach((x, i) => {
      if (i > 0 && (cuts[i] ?? 0) < 0.4) towers.push([])
      ;(towers[towers.length - 1] as string[]).push(x)
    })
    return towers
  }
  const [k1, k2] = split(key, 2) as [Key, Key]
  return { initial: deal(k1), goal: deal(k2) }
}
