/**
 * The pure part of a narrative: the key moments of a page resolved against the tape (steps, event firings), the cursor
 * of the page switcher (a switch keeps the step) and the step-through that moves the cursor from one marked step to
 * the next, crossing from a page to the one after it when a page has no more.
 */

import type { Moment, Narrative, NarrativePage } from './layout.ts'

/** A moment resolved to a step of the tape. */
export interface ResolvedMoment {
  readonly step: number
  readonly label?: string
}

/** Where a reader of a narrative stands: the page, and the viewed step. */
export interface NarrativePosition {
  readonly page: number
  readonly step: number
}

/**
 * The moments of a page as steps of the tape so far, ascending and without repeats (the first label of a step wins).
 * A moment at a step the tape has not reached yet is left out until it has; `end` is the head once the run is over.
 *
 * @param moments - The page's moments (none: no moments).
 * @param stepsOf - The steps at which an event fired, by event name.
 * @param head - The head of the tape.
 * @param done - Whether the run has ended.
 * @returns The resolved moments.
 */
export function resolveMoments(
  moments: readonly Moment[] | undefined,
  stepsOf: (event: string) => readonly number[],
  head: number,
  done: boolean,
): ResolvedMoment[] {
  const found = new Map<number, ResolvedMoment>()
  const add = (step: number, label: string | undefined): void => {
    if (!Number.isFinite(step) || step < 0 || step > head || found.has(step)) return
    found.set(step, label === undefined ? { step } : { step, label })
  }
  for (const m of moments ?? []) {
    if ('at' in m) {
      if (m.at === 'start') add(0, m.label)
      else if (m.at === 'end') {
        if (done) add(head, m.label)
      } else add(m.at, m.label)
      continue
    }
    const steps = stepsOf(m.event)
    if (m.nth === undefined) for (const s of steps) add(s, m.label)
    else {
      const s = m.nth === 'last' ? steps.at(-1) : steps[m.nth]
      if (s !== undefined) add(s, m.label)
    }
  }
  return [...found.values()].sort((a, b) => a.step - b.step)
}

/**
 * The page switcher: the page changes and the step does not (the cursor belongs to the tape, not to a page).
 *
 * @param narrative - The narrative.
 * @param position - Where the reader stands.
 * @param page - The page to go to (clamped to the narrative's pages).
 * @returns The new position.
 */
export function switchPage(narrative: Narrative, position: NarrativePosition, page: number): NarrativePosition {
  const last = Math.max(0, narrative.pages.length - 1)
  return { page: Math.min(Math.max(0, Math.round(page)), last), step: position.step }
}

/**
 * The step-through: the next (or previous) marked step. Within the page it is the next moment after (or before) the
 * viewed step; when the page has none left it is the first (or last) moment of the next (or previous) page that has
 * any; `undefined` at the ends of the narrative.
 *
 * @param pages - The resolved moments of each page, in order.
 * @param position - Where the reader stands.
 * @param direction - `1` for the next moment, `-1` for the previous one.
 * @returns The new position, or `undefined` when there is no further moment.
 */
export function stepThrough(
  pages: readonly (readonly ResolvedMoment[])[],
  position: NarrativePosition,
  direction: 1 | -1,
): NarrativePosition | undefined {
  const here = pages[position.page] ?? []
  const within =
    direction === 1 ? here.find((m) => m.step > position.step) : [...here].reverse().find((m) => m.step < position.step)
  if (within !== undefined) return { page: position.page, step: within.step }
  for (let p = position.page + direction; p >= 0 && p < pages.length; p += direction) {
    const moments = pages[p] ?? []
    const target = direction === 1 ? moments[0] : moments.at(-1)
    if (target !== undefined) return { page: p, step: target.step }
  }
  return undefined
}

/**
 * Which moment of a page the viewed step is at or just past, for a "moment 2 of 4" label.
 *
 * @param moments - The page's resolved moments.
 * @param step - The viewed step.
 * @returns The index of the last moment at or before the step, `-1` before the first.
 */
export function momentIndex(moments: readonly ResolvedMoment[], step: number): number {
  let index = -1
  moments.forEach((m, i) => {
    if (m.step <= step) index = i
  })
  return index
}

/**
 * The paragraphs of a page's text: blocks separated by blank lines.
 *
 * @param page - The page.
 * @returns The paragraphs, trimmed.
 */
export function paragraphs(page: Pick<NarrativePage, 'text'>): string[] {
  return page.text
    .split(/\n\s*\n/)
    .map((p) => p.trim())
    .filter((p) => p !== '')
}

/**
 * The inline pieces of a paragraph: plain text, `code` and **bold**.
 *
 * @param paragraph - The paragraph.
 * @returns The pieces, in order.
 */
export function inlinePieces(paragraph: string): { readonly kind: 'text' | 'code' | 'bold'; readonly text: string }[] {
  const out: { kind: 'text' | 'code' | 'bold'; text: string }[] = []
  const re = /`([^`]+)`|\*\*([^*]+)\*\*/g
  let last = 0
  for (let m = re.exec(paragraph); m !== null; m = re.exec(paragraph)) {
    if (m.index > last) out.push({ kind: 'text', text: paragraph.slice(last, m.index) })
    out.push(m[1] !== undefined ? { kind: 'code', text: m[1] } : { kind: 'bold', text: m[2] ?? '' })
    last = m.index + m[0].length
  }
  if (last < paragraph.length) out.push({ kind: 'text', text: paragraph.slice(last) })
  return out
}
