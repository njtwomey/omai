import { scene } from 'mdml-engine/scene'
import { describe, expect, it } from 'vitest'
import { filmstrip } from './frames.ts'

describe('filmstrip', () => {
  it('lays the frames side by side in rows of the given width', () => {
    const frames = [scene(10, 10, []), scene(10, 10, []), scene(10, 10, [])]
    expect(filmstrip(frames, { columns: 2, cell: [20, 20] }).width).toBe(40)
  })
})
