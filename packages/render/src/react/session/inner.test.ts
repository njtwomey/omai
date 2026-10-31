import { describe, expect, it } from 'vitest'
import { innerCursor } from './InnerPanel.tsx'

describe('the inner cursor', () => {
  it('follows the outer step, held at the end of the inner run', () => {
    expect(innerCursor('outer', 7, 20, null)).toBe(7)
    expect(innerCursor('outer', 70, 20, 3)).toBe(20)
    expect(innerCursor('outer', -2, 20, null)).toBe(0)
  })

  it('moves on its own, at the end of the inner run until moved', () => {
    expect(innerCursor('own', 7, 20, null)).toBe(20)
    expect(innerCursor('own', 7, 20, 3)).toBe(3)
    expect(innerCursor('own', 7, 20, 99)).toBe(20)
    expect(innerCursor('own', 7, 20, 2.6)).toBe(3)
  })
})
