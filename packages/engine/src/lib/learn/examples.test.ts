// Runs every TSDoc @example of the learn domain (and its subfolders) and checks each `// => value` line.
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { stripTypeScriptTypes } from 'node:module'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import * as agents from '../agents/index.ts'
import * as autodiff from '../autodiff/index.ts'
import * as base from '../base/index.ts'
import * as data from '../data/index.ts'
import * as dists from '../dists/index.ts'
import * as exposition from '../exposition/index.ts'
import * as losses from '../losses/index.ts'
import * as metrics from '../metrics/index.ts'
import * as nn from '../nn/index.ts'
import * as optim from '../optim/index.ts'
import * as random from '../random/index.ts'
import * as space from '../space/index.ts'
import * as stats from '../stats/index.ts'
import * as svm from '../svm/index.ts'
import * as tensor from '../tensor/index.ts'
import * as trace from '../trace/index.ts'
import * as learn from './index.ts'

const MODULES: Readonly<Record<string, Readonly<Record<string, unknown>>>> = {
  'omai-engine/agents': agents,
  'omai-engine/autodiff': autodiff,
  'omai-engine/base': base,
  'omai-engine/data': data,
  'omai-engine/dists': dists,
  'omai-engine/exposition': exposition,
  'omai-engine/losses': losses,
  'omai-engine/metrics': metrics,
  'omai-engine/nn': nn,
  'omai-engine/optim': optim,
  'omai-engine/random': random,
  'omai-engine/space': space,
  'omai-engine/stats': stats,
  'omai-engine/svm': svm,
  'omai-engine/tensor': tensor,
  'omai-engine/trace': trace,
  'omai-engine/learn': learn,
  ...Object.fromEntries(
    Object.entries(learn)
      .filter(([, v]) => typeof v === 'object' && v !== null && !Array.isArray(v))
      .flatMap(([k, v]) => {
        const folder = k.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`)
        return [[`omai-engine/learn/${folder}`, v as Readonly<Record<string, unknown>>]]
      }),
  ),
}

interface Example {
  readonly file: string
  readonly line: number
  readonly code: string[]
}

function examples(file: string): Example[] {
  const text = readFileSync(join(import.meta.dirname, file), 'utf8').split('\n')
  const out: Example[] = []
  for (let i = 0; i < text.length; i++) {
    if (!/^\s*\*\s*```ts\s*$/.test(text[i] ?? '')) continue
    const code: string[] = []
    const start = i
    for (i++; i < text.length && !/^\s*\*\s*```\s*$/.test(text[i] ?? ''); i++) {
      code.push((text[i] ?? '').replace(/^\s*\*\s?/, ''))
    }
    out.push({ file, line: start + 1, code })
  }
  return out
}

function compileExample(code: readonly string[]): string {
  const body: string[] = []
  for (const line of code) {
    const imp = /^import\s+\{([^}]*)\}\s+from\s+'([^']+)'/.exec(line)
    if (imp !== null) {
      const names = (imp[1] ?? '').replace(/\btype\s+\w+\s*,?/g, '').replace(/\s+as\s+/g, ': ')
      body.push(`const {${names}} = __mods[${JSON.stringify(imp[2])}];`)
      continue
    }
    const throws = /^(.*?)\s*\/\/ => throws (\w[\w-]*)$/.exec(line)
    if (throws !== null) {
      body.push(`__throws(() => (${throws[1]}), ${JSON.stringify(throws[2])}, ${JSON.stringify(line)});`)
      continue
    }
    const traced = /^(.*?)\s*\/\/ => (trace|figure)$/.exec(line)
    if (traced !== null) {
      body.push(`${traced[1]};`)
      continue
    }
    const check = /^(.*?)\s*\/\/ => (.*)$/.exec(line)
    if (check !== null) {
      body.push(`__check(() => (${check[1]}), () => (${check[2]}), ${JSON.stringify(line)});`)
      continue
    }
    body.push(line)
  }
  // Examples are TypeScript (`as const`, annotations); the function constructor reads JavaScript.
  return stripTypeScriptTypes(body.join('\n'))
}

/**
 * Whether a computed value matches the value an example states. As in the examples runner of `omai-meta`
 * (`runner/compare.ts`), a number written with d decimals matches any value that rounds to it at d decimals; an integer
 * or a longer literal matches to a relative 1e-12. Arrays and plain objects compare element by element; anything else
 * must be equal.
 */
function matches(actual: unknown, expected: unknown): boolean {
  if (typeof actual === 'number' && typeof expected === 'number') {
    if (Object.is(actual, expected) || (Number.isNaN(actual) && Number.isNaN(expected))) return true
    if (!Number.isFinite(actual) || !Number.isFinite(expected)) return false
    const m = /^-?\d+(?:\.(\d+))?(?:e([+-]?\d+))?$/.exec(String(expected))
    const decimals = (m?.[1] ?? '').length
    const exponent = Number(m?.[2] ?? 0)
    const half = decimals === 0 ? 0 : 0.5 * 10 ** (exponent - decimals)
    return Math.abs(actual - expected) <= Math.max(half * 1.000001, 1e-12 * Math.max(1, Math.abs(expected)))
  }
  if (Array.isArray(actual) && Array.isArray(expected))
    return actual.length === expected.length && actual.every((a, i) => matches(a, expected[i]))
  if (
    typeof actual === 'object' &&
    typeof expected === 'object' &&
    actual !== null &&
    expected !== null &&
    Object.getPrototypeOf(actual) === Object.prototype &&
    Object.getPrototypeOf(expected) === Object.prototype
  ) {
    const ka = Object.keys(actual)
    const kb = Object.keys(expected)
    return (
      ka.length === kb.length &&
      ka.every(
        (k) =>
          k in expected && matches((actual as Record<string, unknown>)[k], (expected as Record<string, unknown>)[k]),
      )
    )
  }
  return Object.is(actual, expected) || JSON.stringify(actual) === JSON.stringify(expected)
}

function sources(dir: string, prefix = ''): string[] {
  const out: string[] = []
  for (const name of readdirSync(join(import.meta.dirname, dir)).sort()) {
    const rel = prefix === '' ? name : `${prefix}/${name}`
    if (statSync(join(import.meta.dirname, rel)).isDirectory()) out.push(...sources(rel, rel))
    else if (name.endsWith('.ts') && !name.includes('.test') && !name.startsWith('_') && name !== 'index.ts')
      out.push(rel)
  }
  return out
}

const files = sources('.')

describe('documentation examples', () => {
  for (const file of files) {
    for (const ex of examples(file)) {
      it(`${file}:${ex.line}`, () => {
        const run = new Function('__mods', '__check', '__throws', compileExample(ex.code)) as (
          mods: typeof MODULES,
          check: (actual: () => unknown, expected: () => unknown, text: string) => void,
          throws: (f: () => unknown, code: string, text: string) => void,
        ) => void
        run(
          MODULES,
          (actual, expected, text) => {
            const a = actual()
            const e = expected()
            if (!matches(a, e)) expect(a, text).toEqual(e)
          },
          (f, code, text) => {
            expect(f, text).toThrow()
            try {
              f()
            } catch (e) {
              expect(base.isOmaiError(e, code as base.ErrorCode), text).toBe(true)
            }
          },
        )
      })
    }
  }
})
