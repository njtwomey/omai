/**
 * The translation edit rate of Snover et al. (2006) as Tercom and sacrebleu compute it: greedy block shifts that most
 * reduce a beam-limited word edit distance, then the edit distance of the shifted hypothesis.
 */

import { MdmlError } from '../base/index.ts'

const MAX_SHIFT_SIZE = 10
const MAX_SHIFT_DIST = 50
const BEAM_WIDTH = 25
const MAX_SHIFT_CANDIDATES = 1000
const INFINITY = 1e16

type Op = 'i' | 'd' | ' ' | 's' | 'x'

/** The beam-limited edit distance from a hypothesis to the reference, with its trace of operations. */
function editDistance(hyp: readonly string[], ref: readonly string[]): { distance: number; trace: string } {
  const nh = hyp.length
  const nr = ref.length
  const cost: Float64Array[] = [Float64Array.from({ length: nr + 1 }, (_, j) => j)]
  const ops: Op[][] = [Array.from({ length: nr + 1 }, (): Op => 'i')]
  for (let i = 0; i < nh; i++) {
    cost.push(new Float64Array(nr + 1).fill(INFINITY))
    ops.push(new Array<Op>(nr + 1).fill('x'))
  }
  const ratio = nh > 0 ? nr / nh : 1
  const beam = BEAM_WIDTH < ratio / 2 ? Math.ceil(ratio / 2 + BEAM_WIDTH) : BEAM_WIDTH
  for (let i = 1; i <= nh; i++) {
    const diag = Math.floor(i * ratio)
    const lo = Math.max(0, diag - beam)
    const hi = i === nh ? nr + 1 : Math.min(nr + 1, diag + beam)
    const row = cost[i] as Float64Array
    const prev = cost[i - 1] as Float64Array
    const rowOps = ops[i] as Op[]
    for (let j = lo; j < hi; j++) {
      if (j === 0) {
        row[0] = (prev[0] ?? 0) + 1
        rowOps[0] = 'd'
        continue
      }
      const same = hyp[i - 1] === ref[j - 1]
      const candidates: readonly [number, Op][] = [
        [(prev[j - 1] ?? 0) + (same ? 0 : 1), same ? ' ' : 's'],
        [(prev[j] ?? 0) + 1, 'd'],
        [(row[j - 1] ?? 0) + 1, 'i'],
      ]
      for (const [c, op] of candidates) {
        if ((row[j] ?? 0) > c) {
          row[j] = c
          rowOps[j] = op
        }
      }
    }
  }
  let trace = ''
  let i = nh
  let j = nr
  while (i > 0 || j > 0) {
    const op = ops[i]?.[j] ?? 'x'
    trace = op + trace
    if (op === 's' || op === ' ') {
      i--
      j--
    } else if (op === 'i') j--
    else if (op === 'd') i--
    else throw new MdmlError('value', 'translationEditRate: undefined edit operation')
  }
  return { distance: cost[nh]?.[nr] ?? 0, trace }
}

/** Reads a trace (flipped to rewrite the reference into the hypothesis) as an alignment and error positions. */
function alignment(trace: string): { align: Map<number, number>; refErr: number[]; hypErr: number[] } {
  let ph = -1
  let pr = -1
  const align = new Map<number, number>()
  const refErr: number[] = []
  const hypErr: number[] = []
  for (const op of trace) {
    if (op === ' ' || op === 's') {
      ph++
      pr++
      align.set(pr, ph)
      hypErr.push(op === 's' ? 1 : 0)
      refErr.push(op === 's' ? 1 : 0)
    } else if (op === 'i') {
      ph++
      hypErr.push(1)
    } else {
      pr++
      align.set(pr, ph)
      refErr.push(1)
    }
  }
  return { align, refErr, hypErr }
}

function performShift(words: readonly string[], start: number, length: number, target: number): string[] {
  if (target < start)
    return [
      ...words.slice(0, target),
      ...words.slice(start, start + length),
      ...words.slice(target, start),
      ...words.slice(start + length),
    ]
  if (target > start + length)
    return [
      ...words.slice(0, start),
      ...words.slice(start + length, target),
      ...words.slice(start, start + length),
      ...words.slice(target),
    ]
  return [
    ...words.slice(0, start),
    ...words.slice(start + length, length + target),
    ...words.slice(start, start + length),
    ...words.slice(length + target),
  ]
}

/** Python's ordering of lists of strings. */
function compareWords(a: readonly string[], b: readonly string[]): number {
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    const x = a[i] ?? ''
    const y = b[i] ?? ''
    if (x !== y) return x < y ? -1 : 1
  }
  return a.length - b.length
}

type Candidate = readonly [number, number, number, number, readonly string[]]

function better(a: Candidate, b: Candidate): boolean {
  for (let k = 0; k < 4; k++) if ((a[k] as number) !== (b[k] as number)) return (a[k] as number) > (b[k] as number)
  return compareWords(a[4], b[4]) > 0
}

const sum = (xs: readonly number[]): number => xs.reduce((s, v) => s + v, 0)

/** The best shift of the hypothesis towards the reference, with the number of candidates checked. */
function shift(
  words: readonly string[],
  ref: readonly string[],
  checked: number,
): { gain: number; words: readonly string[]; checked: number } {
  const { distance: pre, trace: inv } = editDistance(words, ref)
  const flipped = Array.from(inv, (c) => (c === 'i' ? 'd' : c === 'd' ? 'i' : c)).join('')
  const { align, refErr, hypErr } = alignment(flipped)
  let best: Candidate | undefined
  let count = checked
  outer: for (let sh = 0; sh < words.length; sh++) {
    for (let sr = 0; sr < ref.length; sr++) {
      if (Math.abs(sr - sh) > MAX_SHIFT_DIST) continue
      let length = 0
      while (words[sh + length] === ref[sr + length] && length < MAX_SHIFT_SIZE) {
        length++
        const ok =
          sum(hypErr.slice(sh, sh + length)) !== 0 &&
          sum(refErr.slice(sr, sr + length)) !== 0 &&
          !(sh <= (align.get(sr) ?? Number.NaN) && (align.get(sr) ?? Number.NaN) < sh + length)
        if (ok) {
          let prevIdx = -1
          for (let offset = -1; offset < length; offset++) {
            let idx: number
            if (sr + offset === -1) idx = 0
            else {
              const a = align.get(sr + offset)
              if (a === undefined) break
              idx = a + 1
            }
            if (idx === prevIdx) continue
            prevIdx = idx
            const shifted = performShift(words, sh, length, idx)
            const candidate: Candidate = [pre - editDistance(shifted, ref).distance, length, -sh, -idx, shifted]
            count++
            if (best === undefined || better(candidate, best)) best = candidate
          }
          if (count >= MAX_SHIFT_CANDIDATES) break outer
        }
        if (words.length === sh + length || ref.length === sr + length) break
      }
    }
  }
  return best === undefined ? { gain: 0, words, checked: count } : { gain: best[0], words: best[4], checked: count }
}

/** The number of edits (shifts plus edit distance) from the hypothesis words to the reference words. */
export function terEdits(hyp: readonly string[], ref: readonly string[]): number {
  if (ref.length === 0) return hyp.length
  let shifts = 0
  let words = hyp
  let checked = 0
  for (;;) {
    const s = shift(words, ref, checked)
    checked = s.checked
    if (checked >= MAX_SHIFT_CANDIDATES || s.gain <= 0) break
    shifts++
    words = s.words
  }
  return shifts + editDistance(words, ref).distance
}
