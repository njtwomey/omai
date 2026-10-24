import { describe, expect, it } from 'vitest'
import { advanceFrame, framesDue, timeStamp } from './playhead.ts'

describe('the playhead', () => {
  it('releases frames at the target rate, carrying the remainder', () => {
    expect(framesDue(16, 10)).toEqual({ frames: 0, carry: 16 })
    const due = framesDue(250, 10)
    expect(due.frames).toBe(2)
    expect(due.carry).toBeCloseTo(50, 9)
    expect(framesDue(10_000, 30)).toEqual({ frames: 4, carry: 0 })
    expect(framesDue(100, 0)).toEqual({ frames: 0, carry: 100 })
  })

  it('wraps with loop and stops at the ends without', () => {
    expect(advanceFrame(9, 1, 10, true)).toEqual({ frame: 0, ended: false })
    expect(advanceFrame(0, -1, 10, true)).toEqual({ frame: 9, ended: false })
    expect(advanceFrame(8, 3, 10, false)).toEqual({ frame: 9, ended: true })
    expect(advanceFrame(0, -1, 10, false)).toEqual({ frame: 0, ended: false })
  })

  it('stamps frames as minutes and seconds', () => {
    expect(timeStamp(0, 10)).toBe('0:00.00')
    expect(timeStamp(25, 10)).toBe('0:02.50')
    expect(timeStamp(615, 10)).toBe('1:01.50')
  })
})
