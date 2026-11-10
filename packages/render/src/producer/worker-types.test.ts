import { describe, expect, it } from 'vitest'
import type { WorkerLike } from './protocol.ts'
import type { StageWorker } from './stage.ts'

// A real `Worker` must be accepted where the producer and the stage ask for the part of it they use, so a host can write
// `createWorker: () => new Worker(new URL(...), { type: 'module' })` without a cast.
describe('worker types', () => {
  it('a Worker is a StageWorker and a WorkerLike', () => {
    const accepts = (_stage: StageWorker, _like: WorkerLike): boolean => true
    // Compile-time only: the function is never called.
    const never = (w: Worker): boolean => accepts(w, w)
    expect(typeof never).toBe('function')
  })
})
