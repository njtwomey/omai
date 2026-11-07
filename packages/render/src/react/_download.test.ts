import { afterEach, describe, expect, it, vi } from 'vitest'
import { saveBlob, saveUrl } from './_download.ts'

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

describe('saving a file', () => {
  it('clicks an anchor with the name, and frees a blob URL once the browser has it', () => {
    vi.useFakeTimers()
    const clicks: { href: string; download: string }[] = []
    vi.stubGlobal('document', {
      createElement: () => {
        const a = { href: '', download: '', click: () => clicks.push({ href: a.href, download: a.download }) }
        return a
      },
    })
    const revoked: string[] = []
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:one')
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation((u) => void revoked.push(u))
    saveUrl('data:text/plain,hi', 'a.txt')
    saveBlob('{}', 'application/json', 'b.json')
    expect(clicks).toEqual([
      { href: 'data:text/plain,hi', download: 'a.txt' },
      { href: 'blob:one', download: 'b.json' },
    ])
    expect(revoked).toEqual([])
    vi.advanceTimersByTime(10_000)
    expect(revoked).toEqual(['blob:one'])
    vi.restoreAllMocks()
  })
})
