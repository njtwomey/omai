/**
 * Lindenmayer systems: a string rewritten in parallel by rules, generation by generation, and read as drawing
 * instructions by a turtle. Koch's curve, the dragon, the Hilbert and Peano space-filling curves, Sierpinski's
 * arrowhead and a branching plant are a line of grammar each; stochastic rules give a different plant from the same
 * grammar on every run. The rewriting, the turtle, the named grammars with the counts and shapes they are known for,
 * and the generation-by-generation algorithm that logs the string's length and the curve it draws.
 */

import { type Json, OmaiError } from '../../base/index.ts'
import { foldIn, key as keyOf } from '../../random/index.ts'
import {
  type Algorithm,
  type Contract,
  contractSchema,
  defineContract,
  derived,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../../trace/index.ts'
import { draws } from '../automata/_core.ts'
import { boxCountingDimension } from './box-counting.ts'

/** What a symbol becomes: a string, or alternatives with chances (a stochastic rule). */
export type LRule = string | readonly { readonly p: number; readonly to: string }[]

/** A grammar and how to draw it. */
export interface LSystem {
  readonly name: string
  readonly axiom: string
  /** Rules by symbol; a symbol without a rule stays as it is. */
  readonly rules: Readonly<Record<string, LRule>>
  /** The turning angle in degrees. */
  readonly angle: number
  /** The symbols that draw a unit segment while moving forward (default `F`); `f` moves without drawing. */
  readonly draw?: string
}

/** The names of {@link L_SYSTEMS}. */
export type LSystemName = 'koch' | 'dragon' | 'hilbert' | 'peano' | 'arrowhead' | 'plant' | 'stochastic-plant'

/**
 * Named grammars. `+` turns left by the angle, `-` right, `[` and `]` save and restore the turtle, `F` draws forward.
 * Known counts: after n generations Koch's curve has 4^n segments and ends 3^n from where it began; the dragon has 2^n
 * segments, none drawn twice; the Hilbert curve has 4^n - 1 segments and visits the 4^n points of a 2^n square grid once
 * each; the Peano curve has 9^n - 1 segments through the 9^n points of a 3^n square grid; the arrowhead has 3^n.
 */
export const L_SYSTEMS: Readonly<Record<LSystemName, LSystem>> = {
  koch: { name: 'koch', axiom: 'F', rules: { F: 'F+F--F+F' }, angle: 60 },
  dragon: { name: 'dragon', axiom: 'FX', rules: { X: 'X+YF+', Y: '-FX-Y' }, angle: 90 },
  hilbert: { name: 'hilbert', axiom: 'A', rules: { A: '+BF-AFA-FB+', B: '-AF+BFB+FA-' }, angle: 90 },
  peano: {
    name: 'peano',
    axiom: 'X',
    rules: { X: 'XFYFX+F+YFXFY-F-XFYFX', Y: 'YFXFY-F-XFYFX+F+YFXFY' },
    angle: 90,
  },
  arrowhead: { name: 'arrowhead', axiom: 'A', rules: { A: 'B-A-B', B: 'A+B+A' }, angle: 60, draw: 'AB' },
  plant: { name: 'plant', axiom: 'X', rules: { X: 'F+[[X]-X]-F[-FX]+X', F: 'FF' }, angle: 25 },
  'stochastic-plant': {
    name: 'stochastic-plant',
    axiom: 'X',
    rules: {
      X: [
        { p: 1 / 3, to: 'F[+X]F[-X]+X' },
        { p: 1 / 3, to: 'F[+X]F[-X]-X' },
        { p: 1 / 3, to: 'F[+X][-X]FX' },
      ],
      F: 'FF',
    },
    angle: 22.5,
  },
}

function checkSystem(name: string, system: LSystem): void {
  for (const [symbol, rule] of Object.entries(system.rules)) {
    if (symbol.length !== 1) throw new OmaiError('value', `${name}: rule symbol '${symbol}' must be one character`)
    if (typeof rule !== 'string') {
      const total = rule.reduce((s, r) => s + r.p, 0)
      if (rule.length === 0 || rule.some((r) => !(r.p >= 0)) || Math.abs(total - 1) > 1e-9)
        throw new OmaiError('value', `${name}: the alternatives for '${symbol}' need non-negative chances summing to 1`)
    }
  }
}

/**
 * One generation of rewriting: every symbol is replaced at once by its rule (stochastic rules draw an alternative per
 * symbol from the uniform numbers given; a symbol with no rule stays).
 *
 * @param text - The current string.
 * @param system - The grammar.
 * @param u - Uniform numbers in [0, 1), one per character of `text` (only used by stochastic rules; may be omitted if none).
 * @returns The next string.
 * @throws OmaiError with code `value` for a rule symbol of more than one character or alternatives whose chances do not sum to 1.
 * @concept l-systems
 * @example
 * ```ts
 * import { fractals } from 'omai-engine/dynamics'
 * fractals.rewriteOnce('F', fractals.L_SYSTEMS.koch) // => 'F+F--F+F'
 * ```
 */
export function rewriteOnce(text: string, system: LSystem, u?: ArrayLike<number>): string {
  checkSystem('rewriteOnce', system)
  const out: string[] = []
  for (let i = 0; i < text.length; i++) {
    const ch = text[i] as string
    const rule = system.rules[ch]
    if (rule === undefined) out.push(ch)
    else if (typeof rule === 'string') out.push(rule)
    else {
      let r = u?.[i] ?? 0
      let chosen = rule[rule.length - 1] as { to: string }
      for (const alt of rule) {
        if (r < alt.p) {
          chosen = alt
          break
        }
        r -= alt.p
      }
      out.push(chosen.to)
    }
  }
  return out.join('')
}

/**
 * The string after some generations of a grammar.
 *
 * @param system - The grammar.
 * @param generations - How many.
 * @param seed - The seed for stochastic rules (default 0).
 * @returns The string.
 * @throws OmaiError with code `value` for negative generations.
 * @concept l-systems
 * @example
 * ```ts
 * import { fractals } from 'omai-engine/dynamics'
 * fractals.rewrite(fractals.L_SYSTEMS.dragon, 2) // => 'FX+YF++-FX-YF+'
 * ```
 */
export function rewrite(system: LSystem, generations: number, seed = 0): string {
  if (!(Number.isInteger(generations) && generations >= 0))
    throw new OmaiError('value', `rewrite: bad generations ${generations}`)
  let text = system.axiom
  for (let g = 0; g < generations; g++) text = rewriteOnce(text, system, draws(foldIn(keyOf(seed), g), text.length))
  return text
}

/** The drawing of a string. */
export interface Turtle {
  /** The segments drawn, flat `[x0, y0, x1, y1, ...]`. */
  readonly segments: Float64Array
  /** The bounding box, `[xmin, xmax, ymin, ymax]`. */
  readonly bounds: readonly [number, number, number, number]
}

/**
 * Draw a string with a turtle that starts at the origin heading along +x: a symbol of `draw` (default `F`) moves forward
 * `step` and draws, `f` moves without drawing, `+` turns left and `-` right by `angle` degrees, `|` turns round, `[`
 * saves position and heading and `]` returns to them; any other symbol does nothing.
 *
 * @param text - The string.
 * @param angle - The turning angle in degrees.
 * @param options - The step length (default 1) and the drawing symbols (default `F`).
 * @returns The segments and their bounds.
 * @throws OmaiError with code `value` for a `]` with no `[`.
 * @concept l-systems
 * @example
 * ```ts
 * import { fractals } from 'omai-engine/dynamics'
 * Array.from(fractals.turtle('F+F', 90).segments).map(Math.round) // => [0, 0, 1, 0, 1, 0, 1, 1]
 * ```
 */
export function turtle(
  text: string,
  angle: number,
  options: { readonly step?: number; readonly draw?: string } = {},
): Turtle {
  const step = options.step ?? 1
  const draws_ = options.draw ?? 'F'
  const turn = (angle * Math.PI) / 180
  let x = 0
  let y = 0
  let h = 0
  const stack: [number, number, number][] = []
  const segments: number[] = []
  let xmin = 0
  let xmax = 0
  let ymin = 0
  let ymax = 0
  for (const ch of text) {
    if (draws_.includes(ch) || ch === 'f') {
      const nx = x + step * Math.cos(h)
      const ny = y + step * Math.sin(h)
      if (ch !== 'f') segments.push(x, y, nx, ny)
      x = nx
      y = ny
      xmin = Math.min(xmin, x)
      xmax = Math.max(xmax, x)
      ymin = Math.min(ymin, y)
      ymax = Math.max(ymax, y)
    } else if (ch === '+') h += turn
    else if (ch === '-') h -= turn
    else if (ch === '|') h += Math.PI
    else if (ch === '[') stack.push([x, y, h])
    else if (ch === ']') {
      const top = stack.pop()
      if (top === undefined) throw new OmaiError('value', 'turtle: a ] with no matching [')
      ;[x, y, h] = top
    }
  }
  return { segments: Float64Array.from(segments), bounds: [xmin, xmax, ymin, ymax] }
}

/** Options of {@link lSystemAlgorithm}. */
export interface LSystemOptions {
  /** The grammar (default Koch's curve). */
  readonly system?: LSystem
  /** The most characters a generation may have before the run is refused (default 4 million). */
  readonly maxLength?: number
}

/** The state of {@link lSystemAlgorithm}. */
export interface LSystemState {
  readonly text: string
  readonly generation: number
}

/** The paths of {@link lSystemAlgorithm}. */
export type LSystemPath =
  | 'generation/n'
  | 'generation/length'
  | 'generation/string'
  | 'generation/curve'
  | 'curve/dimension'

/** The log contract of {@link lSystemAlgorithm}. `generation/string` is derived: the first 200 characters. */
export const lSystemContract: Contract<LSystemState, LSystemPath> = defineContract({
  'generation/n': scalar({ description: 'the generation' }),
  'generation/length': scalar({ description: 'characters in the string' }),
  'generation/string': derived<LSystemState>({
    description: 'the first 200 characters of the string',
    compute: (s) => s.text.slice(0, 200),
  }),
  'generation/curve': tensor({
    role: 'state',
    shape: ['segments', 4],
    description: 'the drawn segments x0, y0, x1, y1',
  }),
  'curve/dimension': scalar({
    description: "the box-counting dimension of the curve's vertices (NaN until there are enough)",
  }),
})

/**
 * An L-system as a traced algorithm: one generation of rewriting per step, the string's length and the turtle's curve
 * (unit steps) on the tape, and the box-counting dimension of the curve's vertices once there are enough. Stochastic
 * rules draw their alternatives from the step's key, so a seed is a plant. The state serialises.
 *
 * @param options - The grammar and the length limit.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for a malformed grammar, or when a generation would exceed `maxLength`.
 * @concept l-systems
 * @example
 * ```ts
 * import { fractals } from 'omai-engine/dynamics'
 * import { runAlgorithm } from 'omai-engine/trace'
 * runAlgorithm(fractals.lSystemAlgorithm(), { seed: 0, maxSteps: 3 }).state.text.length // => 148
 * ```
 */
export function lSystemAlgorithm(options: LSystemOptions = {}): Algorithm<LSystemState, LSystemPath> {
  const system = options.system ?? L_SYSTEMS.koch
  const maxLength = options.maxLength ?? 4_000_000
  checkSystem('lSystemAlgorithm', system)
  return {
    name: 'dynamics/fractals/l-system',
    config: {
      system: system.name,
      axiom: system.axiom,
      rules: system.rules as unknown as Json,
      angle: system.angle,
      maxLength,
    },
    schema: contractSchema(lSystemContract),
    init: () => ({ text: system.axiom, generation: 0 }),
    step: (s, ctx) => {
      const text = rewriteOnce(s.text, system, draws(foldIn(ctx.key, 0), s.text.length))
      if (text.length > maxLength)
        throw new OmaiError(
          'value',
          `lSystemAlgorithm: generation ${s.generation + 1} has more than ${maxLength} characters`,
        )
      return { text, generation: s.generation + 1 }
    },
    observe: (s) => {
      const t = turtle(s.text, system.angle, system.draw === undefined ? {} : { draw: system.draw })
      let dimension = Number.NaN
      if (t.segments.length >= 4 * 64) {
        const ends = new Float64Array(t.segments.length / 2)
        for (let i = 0; i < ends.length / 2; i++) {
          ends[2 * i] = t.segments[4 * i + 2] as number
          ends[2 * i + 1] = t.segments[4 * i + 3] as number
        }
        dimension = boxCountingDimension(ends, { minLevel: 1 }).dimension
      }
      return {
        metrics: { 'generation/n': s.generation, 'generation/length': s.text.length, 'curve/dimension': dimension },
        tensors: { 'generation/curve': { role: 'state', value: t.segments, shape: [t.segments.length / 4, 4] } },
      }
    },
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as LSystemState,
  }
}
