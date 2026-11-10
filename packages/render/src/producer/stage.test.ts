import { describe, expect, it } from 'vitest'
import { type StageContext, type StageScope, type StageWorker, serveStage, workerStage } from './stage.ts'

// A worker the test drives by hand.
class FakeWorker implements StageWorker {
  readonly sent: unknown[] = []
  terminated = 0
  onmessage: StageWorker['onmessage'] = null
  onerror: StageWorker['onerror'] = null
  postMessage(message: unknown): void {
    this.sent.push(message)
  }
  terminate(): void {
    this.terminated++
  }
  say(data: unknown): void {
    this.onmessage?.({ data })
  }
}

// What a job gives a stage: progress is recorded, cleanups run on `cancel`.
function context(): {
  readonly ctx: StageContext
  readonly progress: number[]
  cancel(): void
  readonly pending: () => number
} {
  const progress: number[] = []
  const undo = new Set<() => void>()
  return {
    progress,
    pending: () => undo.size,
    cancel: () => {
      for (const u of [...undo]) u()
    },
    ctx: {
      progress: (f) => progress.push(f),
      onCancel: (u) => {
        undo.add(u)
        return () => void undo.delete(u)
      },
    },
  }
}

describe('workerStage', () => {
  it('posts the input made from the previous result, relays progress, resolves with the result and terminates', async () => {
    const w = new FakeWorker()
    const stage = workerStage<number, string>({
      name: 'solve',
      weight: 3,
      createWorker: () => w,
      input: (n) => ({ n }),
    })
    expect([stage.name, stage.weight]).toEqual(['solve', 3])
    const c = context()
    const out = stage.run(5, c.ctx)
    expect(w.sent).toEqual([{ type: 'run', input: { n: 5 } }])
    w.say({ type: 'progress', fraction: 0.25 })
    w.say({ type: 'progress', fraction: 0.75 })
    w.say({ type: 'result', value: 'done' })
    expect(await out).toBe('done')
    expect(c.progress).toEqual([0.25, 0.75])
    expect(w.terminated).toBe(1)
    // The cleanup it registered is taken back: a later cancel does not terminate again.
    expect(c.pending()).toBe(0)
    c.cancel()
    expect(w.terminated).toBe(1)
  })

  it('sends the previous result itself without an `input`, and ignores messages it does not know', async () => {
    const w = new FakeWorker()
    const c = context()
    const out = workerStage<number[]>({ name: 's', createWorker: () => w }).run([1, 2], c.ctx)
    w.say('noise')
    w.say({ type: 'other' })
    expect(w.sent).toEqual([{ type: 'run', input: [1, 2] }])
    w.say({ type: 'result', value: 1 })
    expect(await out).toBe(1)
  })

  it('rejects with the message of an error answer or the worker error event, terminating the worker', async () => {
    const a = new FakeWorker()
    const first = workerStage({ name: 's', createWorker: () => a }).run(undefined, context().ctx)
    a.say({ type: 'error', message: 'singular' })
    await expect(first).rejects.toThrow('singular')
    expect(a.terminated).toBe(1)
    const b = new FakeWorker()
    const second = workerStage({ name: 's', createWorker: () => b }).run(undefined, context().ctx)
    b.onerror?.({ message: 'boom' })
    await expect(second).rejects.toThrow('boom')
    expect(b.terminated).toBe(1)
  })

  it('rejects when the input cannot be posted', async () => {
    const w = new FakeWorker()
    w.postMessage = () => {
      throw new Error('not cloneable')
    }
    await expect(workerStage({ name: 's', createWorker: () => w }).run(undefined, context().ctx)).rejects.toThrow(
      'not cloneable',
    )
    expect(w.terminated).toBe(1)
  })

  it('terminates the worker at once when the job is cancelled, and a late answer changes nothing', () => {
    const w = new FakeWorker()
    const c = context()
    let settled = false
    void workerStage({ name: 's', createWorker: () => w })
      .run(undefined, c.ctx)
      .then(
        () => (settled = true),
        () => (settled = true),
      )
    expect(w.terminated).toBe(0)
    c.cancel()
    expect(w.terminated).toBe(1)
    w.say({ type: 'progress', fraction: 0.5 })
    w.say({ type: 'result', value: 1 })
    expect(c.progress).toEqual([])
    expect(w.terminated).toBe(1)
    expect(settled).toBe(false)
  })
})

describe('serveStage', () => {
  // The page's worker and the worker's scope, wired to each other in memory.
  function pair(): { readonly worker: FakeWorker; readonly scope: StageScope } {
    const worker = new FakeWorker()
    const scope: StageScope = {
      onmessage: null,
      postMessage: (m) => queueMicrotask(() => worker.say(m)),
    }
    worker.postMessage = (m) => scope.onmessage?.({ data: m })
    return { worker, scope }
  }

  it('answers a run with progress and the result, end to end with workerStage', async () => {
    const { worker, scope } = pair()
    serveStage(scope, (n: number, progress) => {
      progress(0.5)
      return n * 2
    })
    const c = context()
    expect(await workerStage<number, number>({ name: 's', createWorker: () => worker }).run(21, c.ctx)).toBe(42)
    expect(c.progress).toEqual([0.5])
  })

  it('awaits a promise and reports what was thrown as an error answer', async () => {
    const { worker, scope } = pair()
    serveStage(scope, async (n: number) => {
      if (n < 0) throw new Error('negative')
      return n
    })
    const stage = workerStage<number, number>({ name: 's', createWorker: () => worker })
    expect(await stage.run(2, context().ctx)).toBe(2)
    await expect(stage.run(-1, context().ctx)).rejects.toThrow('negative')
  })
})
