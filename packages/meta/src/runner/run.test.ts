import { describe, expect, it } from 'vitest'
import { runExample, runExamples } from './run.ts'

class CodedError extends Error {
  readonly code: string
  constructor(code: string) {
    super(`failed with ${code}`)
    this.code = code
  }
}

const modules = {
  demo: {
    double: (x: number) => 2 * x,
    fail: (code: string) => {
      throw new CodedError(code)
    },
    later: async (x: number) => x + 1,
    scene: { kind: 'scene' },
  },
}

describe('runExample', () => {
  it('reports each expectation with the value it gave', async () => {
    const r = await runExample(
      "import { double } from 'demo'\nconst n: number = 2\ndouble(n) // => 4\ndouble(0.1) // => 0.3",
      { modules },
    )
    expect(r.ok).toBe(false)
    expect(r.expectations.map((e) => [e.status, e.actual])).toEqual([
      ['pass', '4'],
      ['fail', '0.2'],
    ])
  })

  it('checks thrown error codes, awaits, figures and console output', async () => {
    const code = [
      "import { fail, later, scene } from 'demo'",
      "fail('shape') // => throws shape",
      "fail('value') // => throws shape",
      '(await later(1)) // => 2',
      'scene // => figure',
      "console.log('n =', [1, 2])",
    ].join('\n')
    const r = await runExample(code, { modules, isFigure: (v) => (v as { kind?: string }).kind === 'scene' })
    expect(r.expectations.map((e) => e.status)).toEqual(['pass', 'fail', 'pass', 'pass'])
    expect(r.logs).toEqual(['n = [1, 2]'])
  })

  it('maps a runtime error to its example line and skips what follows', async () => {
    const r = await runExample('const a = 1\nnull.x\na // => 1', {})
    expect(r.error?.line).toBe(1)
    expect(r.expectations[0]?.status).toBe('skipped')
  })

  it('reports syntax errors and unknown modules without throwing', async () => {
    expect((await runExample('const x = (', {})).error?.line).toBe(0)
    expect((await runExample("import { a } from 'nowhere'\na // => 1", {})).error?.message).toContain('nowhere')
  })

  it('reports an expected value that throws as an error', async () => {
    const r = await runExample('1 // => missing.value', {})
    expect(r.expectations[0]?.status).toBe('error')
  })

  it('bounds asynchronous waits', async () => {
    const r = await runExample('await new Promise(() => {})', { timeoutMs: 20 })
    expect(r.error?.message).toContain('timed out')
  })

  it('counts a list of examples', async () => {
    const report = await runExamples([
      { title: '', code: '1 + 1 // => 2' },
      { title: '', code: '1 // => 2' },
    ])
    expect([report.total, report.passed, report.failed]).toEqual([2, 1, 1])
  })
})
