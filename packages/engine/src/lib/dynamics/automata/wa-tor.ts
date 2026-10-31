/**
 * Wa-Tor, Dewdney's toroidal ocean of fish and sharks: fish swim into empty neighbouring water and breed when old
 * enough; sharks hunt the fish next to them (or swim), grow hungry, starve and breed. The two populations oscillate out
 * of phase, the discrete, spatial cousin of the Lotka-Volterra cycle, and on a small ocean one of them can die out.
 * The model as a traced algorithm with the fish and shark counts and an extinction event.
 */

import { MdmlError } from '../../base/index.ts'
import { foldIn, permutation } from '../../random/index.ts'
import {
  type Algorithm,
  type Contract,
  contractSchema,
  defineContract,
  event,
  scalar,
  tensor,
} from '../../trace/index.ts'
import { codec, draws, mod, positiveInt, unit } from './_core.ts'

/** Options of {@link waTor}. */
export interface WaTorOptions {
  /** Columns and rows of the torus (default 60 by 60). */
  readonly width?: number
  readonly height?: number
  /** The share of the ocean that starts as fish (default 0.3) and as sharks (default 0.01). */
  readonly fish?: number
  readonly sharks?: number
  /** Steps of life before a fish (default 2) or a shark (default 12) breeds. */
  readonly fishBreed?: number
  readonly sharkBreed?: number
  /** The energy a shark starts with and a newborn gets (default 3); it loses one a step and starves at 0. */
  readonly sharkEnergy?: number
  /** The energy a shark gains by eating a fish (default 2). */
  readonly fishEnergy?: number
}

/** The state of {@link waTor}. */
export interface WaTorState {
  /** 0 water, 1 fish, 2 shark, row-major. */
  readonly kind: Uint8Array
  /** Steps lived since birth or the last breeding. */
  readonly age: Uint8Array
  /** A shark's energy (0 for others). */
  readonly energy: Int16Array
  readonly fish: number
  readonly sharks: number
}

/** The paths of {@link waTor}. */
export type WaTorPath = 'world/board' | 'world/fish' | 'world/sharks' | 'world/extinct'

/** The log contract of {@link waTor}. */
export const waTorContract: Contract<WaTorState, WaTorPath> = defineContract({
  'world/board': tensor({ role: 'state', shape: ['height', 'width'], description: '0 water, 1 fish, 2 shark' }),
  'world/fish': scalar({ description: 'fish alive' }),
  'world/sharks': scalar({ description: 'sharks alive' }),
  'world/extinct': event({ context: { species: 'fish or sharks', step: 'when the last one died' } }),
})

/**
 * Wa-Tor as a traced algorithm: one chronon per step. The animals act in a random order, each once: a fish moves to a
 * random empty neighbour (four-neighbourhood on the torus); a shark first loses an energy (and dies at 0), then eats a
 * random neighbouring fish if there is one (gaining energy) or else moves to a random empty neighbour. An animal that
 * has moved and is old enough to breed leaves a newborn behind. Fish and sharks never exceed the sites of the ocean and
 * a shark with no fish to eat starves in exactly `sharkEnergy` steps. `world/extinct` fires once for each species that
 * dies out. The state serialises.
 *
 * @param options - Size, starting shares, breeding ages and shark energies.
 * @returns The algorithm.
 * @throws MdmlError with code `value` for a bad size, shares above 1 in total, or non-positive ages and energies.
 * @concept wa-tor
 * @example
 * ```ts
 * import { automata } from 'mdml-engine/dynamics'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * // Fish alone, breeding every step, double at first: 1 fish, 2, 4.
 * const alg = automata.waTor({ width: 30, height: 30, fish: 1 / 900, sharks: 0, fishBreed: 1 })
 * runAlgorithm(alg, { seed: 0, maxSteps: 2 }).state.fish // => 4
 * ```
 */
export function waTor(options: WaTorOptions = {}): Algorithm<WaTorState, WaTorPath> {
  const width = options.width ?? 60
  const height = options.height ?? 60
  const fishShare = options.fish ?? 0.3
  const sharkShare = options.sharks ?? 0.01
  const fishBreed = options.fishBreed ?? 2
  const sharkBreed = options.sharkBreed ?? 12
  const sharkEnergy = options.sharkEnergy ?? 3
  const fishEnergy = options.fishEnergy ?? 2
  positiveInt('waTor', 'width', width)
  positiveInt('waTor', 'height', height)
  positiveInt('waTor', 'fishBreed', fishBreed)
  positiveInt('waTor', 'sharkBreed', sharkBreed)
  positiveInt('waTor', 'sharkEnergy', sharkEnergy)
  positiveInt('waTor', 'fishEnergy', fishEnergy)
  unit('waTor', 'fish', fishShare)
  unit('waTor', 'sharks', sharkShare)
  if (fishShare + sharkShare > 1)
    throw new MdmlError('value', 'waTor: the fish and sharks cannot fill more than the ocean')
  const n = width * height
  const around = (i: number): number[] => {
    const r = Math.floor(i / width)
    const c = i % width
    return [
      mod(r - 1, height) * width + c,
      mod(r + 1, height) * width + c,
      r * width + mod(c - 1, width),
      r * width + mod(c + 1, width),
    ]
  }
  const count = (kind: Uint8Array, k: number): number => {
    let m = 0
    for (const x of kind) if (x === k) m++
    return m
  }
  return {
    name: 'dynamics/automata/wa-tor',
    config: { width, height, fish: fishShare, sharks: sharkShare, fishBreed, sharkBreed, sharkEnergy, fishEnergy },
    schema: contractSchema(waTorContract),
    init: (key) => {
      const nFish = Math.max(Math.round(fishShare * n), fishShare > 0 ? 1 : 0)
      const nSharks = Math.round(sharkShare * n)
      const order = permutation(key, n).data
      const kind = new Uint8Array(n)
      const age = new Uint8Array(n)
      const energy = new Int16Array(n)
      const u = draws(foldIn(key, 1), n)
      for (let k = 0; k < nFish + nSharks && k < n; k++) {
        const site = order[k] ?? 0
        const shark = k >= nFish
        kind[site] = shark ? 2 : 1
        // Fish start at random ages so the shoal does not breed in step; sharks start newborn and fed.
        age[site] = shark ? 0 : Math.floor((u[site] ?? 0) * fishBreed)
        energy[site] = shark ? sharkEnergy : 0
      }
      return { kind, age, energy, fish: nFish, sharks: nSharks }
    },
    step: (s, ctx) => {
      const kind = Uint8Array.from(s.kind)
      const age = Uint8Array.from(s.age)
      const energy = Int16Array.from(s.energy)
      const acted = new Uint8Array(n)
      const order = permutation(foldIn(ctx.key, 0), n).data
      const u = draws(foldIn(ctx.key, 1), 2 * n)
      for (let o = 0; o < n; o++) {
        const i = order[o] ?? 0
        const k = kind[i] ?? 0
        if (k === 0 || acted[i] === 1) continue
        if (k === 2) {
          energy[i] = (energy[i] ?? 0) - 1
          if ((energy[i] ?? 0) <= 0) {
            kind[i] = 0
            age[i] = 0
            energy[i] = 0
            continue
          }
        }
        age[i] = Math.min(255, (age[i] ?? 0) + 1)
        const nb = around(i)
        const prey = k === 2 ? nb.filter((j) => kind[j] === 1) : []
        const empty = nb.filter((j) => kind[j] === 0)
        const pool = prey.length > 0 ? prey : empty
        if (pool.length === 0) {
          acted[i] = 1
          continue
        }
        const to = pool[Math.min(pool.length - 1, Math.floor((u[2 * i] ?? 0) * pool.length))] as number
        if (prey.length > 0) energy[i] = (energy[i] ?? 0) + fishEnergy
        kind[to] = k
        age[to] = age[i] ?? 0
        energy[to] = energy[i] ?? 0
        acted[to] = 1
        kind[i] = 0
        age[i] = 0
        energy[i] = 0
        const breedAt = k === 1 ? fishBreed : sharkBreed
        if ((age[to] ?? 0) >= breedAt) {
          age[to] = 0
          kind[i] = k
          energy[i] = k === 2 ? sharkEnergy : 0
          acted[i] = 1
        }
      }
      const fish = count(kind, 1)
      const sharks = count(kind, 2)
      if (s.fish > 0 && fish === 0) ctx.emit('world/extinct', { species: 'fish', step: ctx.t + 1 })
      if (s.sharks > 0 && sharks === 0) ctx.emit('world/extinct', { species: 'sharks', step: ctx.t + 1 })
      return { kind, age, energy, fish, sharks }
    },
    observe: (s) => ({
      metrics: { 'world/fish': s.fish, 'world/sharks': s.sharks },
      tensors: { 'world/board': { role: 'state', value: s.kind, shape: [height, width] } },
    }),
    ...codec<WaTorState>(),
  }
}
