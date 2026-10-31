import { describe, expect, it } from 'vitest'
import type { Narrative } from './layout.ts'
import {
  inlinePieces,
  momentIndex,
  paragraphs,
  type ResolvedMoment,
  resolveMoments,
  stepThrough,
  switchPage,
} from './narrative.ts'

const fired: Record<string, number[]> = { 'lap/end': [9, 19, 29, 39] }
const stepsOf = (e: string): readonly number[] => fired[e] ?? []

describe('resolveMoments', () => {
  it('turns steps, starts, ends and event firings into ascending steps', () => {
    const got = resolveMoments(
      [{ at: 25, label: 'late' }, { at: 'start' }, { event: 'lap/end', label: 'a lap' }, { at: 'end', label: 'done' }],
      stepsOf,
      40,
      true,
    )
    expect(got).toEqual([
      { step: 0 },
      { step: 9, label: 'a lap' },
      { step: 19, label: 'a lap' },
      { step: 25, label: 'late' },
      { step: 29, label: 'a lap' },
      { step: 39, label: 'a lap' },
      { step: 40, label: 'done' },
    ])
  })

  it('picks one firing, counted from 0 or the last', () => {
    expect(resolveMoments([{ event: 'lap/end', nth: 1 }], stepsOf, 40, false)).toEqual([{ step: 19 }])
    expect(resolveMoments([{ event: 'lap/end', nth: 'last' }], stepsOf, 40, false)).toEqual([{ step: 39 }])
    expect(resolveMoments([{ event: 'lap/end', nth: 9 }], stepsOf, 40, false)).toEqual([])
  })

  it('leaves out what the tape has not reached, and the end of a run that goes on', () => {
    expect(resolveMoments([{ at: 50 }, { at: 'end' }, { at: 12 }], stepsOf, 30, false)).toEqual([{ step: 12 }])
    expect(
      resolveMoments(
        [
          { at: 30, label: 'x' },
          { at: 30, label: 'y' },
        ],
        stepsOf,
        30,
        false,
      ),
    ).toEqual([{ step: 30, label: 'x' }])
    expect(resolveMoments(undefined, stepsOf, 30, true)).toEqual([])
  })
})

describe('the page switcher', () => {
  const narrative: Narrative = { pages: ['a', 'b', 'c'].map((id) => ({ id, title: id, text: '', layout: [] })) }

  it('keeps the cursor when the page changes, and stays within the pages', () => {
    expect(switchPage(narrative, { page: 0, step: 17 }, 2)).toEqual({ page: 2, step: 17 })
    expect(switchPage(narrative, { page: 2, step: 3 }, 9)).toEqual({ page: 2, step: 3 })
    expect(switchPage(narrative, { page: 1, step: 3 }, -4)).toEqual({ page: 0, step: 3 })
  })
})

describe('the step-through', () => {
  const m = (...steps: number[]): ResolvedMoment[] => steps.map((step) => ({ step }))
  const pages = [m(0, 10, 20), [], m(5, 15), m(30)]

  it('goes to the next marked step of the page, then on to the next page that has any', () => {
    expect(stepThrough(pages, { page: 0, step: 0 }, 1)).toEqual({ page: 0, step: 10 })
    expect(stepThrough(pages, { page: 0, step: 12 }, 1)).toEqual({ page: 0, step: 20 })
    // Past the last of page 0: the empty page is skipped, and the next page starts at its first moment even if earlier.
    expect(stepThrough(pages, { page: 0, step: 20 }, 1)).toEqual({ page: 2, step: 5 })
    expect(stepThrough(pages, { page: 2, step: 15 }, 1)).toEqual({ page: 3, step: 30 })
    expect(stepThrough(pages, { page: 3, step: 30 }, 1)).toBeUndefined()
  })

  it('goes back the same way', () => {
    expect(stepThrough(pages, { page: 0, step: 20 }, -1)).toEqual({ page: 0, step: 10 })
    expect(stepThrough(pages, { page: 2, step: 5 }, -1)).toEqual({ page: 0, step: 20 })
    expect(stepThrough(pages, { page: 3, step: 30 }, -1)).toEqual({ page: 2, step: 15 })
    expect(stepThrough(pages, { page: 0, step: 0 }, -1)).toBeUndefined()
  })

  it('starts from wherever the cursor is, between moments', () => {
    expect(stepThrough(pages, { page: 0, step: 7 }, -1)).toEqual({ page: 0, step: 0 })
    expect(stepThrough(pages, { page: 1, step: 100 }, 1)).toEqual({ page: 2, step: 5 })
    expect(stepThrough(pages, { page: 1, step: 100 }, -1)).toEqual({ page: 0, step: 20 })
  })

  it('says which moment the cursor is at', () => {
    expect(momentIndex(m(0, 10, 20), -1)).toBe(-1)
    expect(momentIndex(m(5, 10, 20), 4)).toBe(-1)
    expect(momentIndex(m(5, 10, 20), 10)).toBe(1)
    expect(momentIndex(m(5, 10, 20), 99)).toBe(2)
  })
})

describe('the text of a page', () => {
  it('splits paragraphs at blank lines and marks code and bold', () => {
    expect(paragraphs({ text: '  one\ntwo\n\n\nthree  ' })).toEqual(['one\ntwo', 'three'])
    expect(inlinePieces('see `n` and **this** now')).toEqual([
      { kind: 'text', text: 'see ' },
      { kind: 'code', text: 'n' },
      { kind: 'text', text: ' and ' },
      { kind: 'bold', text: 'this' },
      { kind: 'text', text: ' now' },
    ])
  })
})
