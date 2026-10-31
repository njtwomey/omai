import { describe, expect, it } from 'vitest'
import { sameSettings, settingsReducer } from './settings.ts'

describe('sameSettings', () => {
  it('compares plain data structurally, ignoring key order', () => {
    expect(sameSettings({ a: 1, b: { c: [1, 2, { d: 'x' }] } }, { b: { c: [1, 2, { d: 'x' }] }, a: 1 })).toBe(true)
    expect(sameSettings({ a: 1 }, { a: 1, b: undefined })).toBe(false)
    expect(sameSettings([1, 2], [1, 2, 3])).toBe(false)
    expect(sameSettings({ a: [] }, { a: {} })).toBe(false)
    expect(sameSettings(null, {})).toBe(false)
  })

  it('treats NaN as equal to itself, so a draft holding one is not forever changed', () => {
    expect(sameSettings({ x: Number.NaN }, { x: Number.NaN })).toBe(true)
    expect(sameSettings(0, -0)).toBe(false)
  })
})

describe('settingsReducer', () => {
  const closed = { open: false, draft: 1 }

  it('opens on a fresh copy of the applied value, so a cancelled edit never comes back', () => {
    const open = settingsReducer(closed, { type: 'open', value: 5 })
    const edited = settingsReducer(open, { type: 'edit', draft: 9 })
    const cancelled = settingsReducer(edited, { type: 'cancel' })
    expect(cancelled.open).toBe(false)
    expect(settingsReducer(cancelled, { type: 'open', value: 5 })).toEqual({ open: true, draft: 5 })
  })

  it('keeps the draft in the state it was in when applied, so the caller can hand it over', () => {
    const edited = settingsReducer(settingsReducer(closed, { type: 'open', value: 5 }), { type: 'edit', draft: 9 })
    expect(settingsReducer(edited, { type: 'apply' })).toEqual({ open: false, draft: 9 })
  })

  it('ignores edits and resets while closed', () => {
    expect(settingsReducer(closed, { type: 'edit', draft: 3 })).toBe(closed)
    expect(settingsReducer(closed, { type: 'reset', defaults: 3 })).toBe(closed)
  })

  it('resets the draft to the defaults without applying or closing', () => {
    const open = settingsReducer(closed, { type: 'open', value: 5 })
    expect(settingsReducer(open, { type: 'reset', defaults: 0 })).toEqual({ open: true, draft: 0 })
  })

  it('lets a press on the backdrop close the dialog only while nothing has been edited', () => {
    const open = settingsReducer(closed, { type: 'open', value: 5 })
    expect(settingsReducer(open, { type: 'backdrop', dirty: false }).open).toBe(false)
    const edited = settingsReducer(open, { type: 'edit', draft: 6 })
    expect(settingsReducer(edited, { type: 'backdrop', dirty: true }).open).toBe(true)
  })
})
