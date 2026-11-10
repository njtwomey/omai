// Runs every TSDoc @example of the probprog domain (all folders) and checks each `// => value` line: numbers agree to
// about nine significant digits, arrays and objects element by element, `// => throws Code` checks the error code and
// `// => trace` only runs the expression.
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { stripTypeScriptTypes } from 'node:module'
import { join, relative } from 'node:path'
import { describe, expect, it } from 'vitest'
import * as autodiff from '../autodiff/index.ts'
import * as base from '../base/index.ts'
import * as data from '../data/index.ts'
import * as dists from '../dists/index.ts'
import * as messagePassing from '../message-passing/index.ts'
import * as linalg from '../numerics/linalg/index.ts'
import * as optim from '../optim/index.ts'
import * as pgm from '../pgm/index.ts'
import * as random from '../random/index.ts'
import * as sampling from '../sampling/index.ts'
import * as scene from '../scene/index.ts'
import * as space from '../space/index.ts'
import * as stats from '../stats/index.ts'
import * as tensor from '../tensor/index.ts'
import * as trace from '../trace/index.ts'
import * as probprog from './index.ts'

const MODULES: Readonly<Record<string, Readonly<Record<string, unknown>>>> = {
  'omai-engine/probprog': probprog,
  'omai-engine/probprog/model': probprog.model,
  'omai-engine/probprog/model/dists': probprog.model.dists,
  'omai-engine/probprog/inference': probprog.inference,
  'omai-engine/probprog/validation': probprog.validation,
  'omai-engine/message-passing': messagePassing,
  'omai-engine/space': space,
  'omai-engine/sampling': sampling,
  'omai-engine/pgm': pgm,
  'omai-engine/optim': optim,
  'omai-engine/data': data,
  'omai-engine/scene': scene,
  'omai-engine/stats': stats,
  'omai-engine/dists/compose': dists.compose,
  'omai-engine/tensor': tensor,
  'omai-engine/random': random,
  'omai-engine/dists': dists,
  'omai-engine/trace': trace,
  'omai-engine/autodiff': autodiff,
  'omai-engine/base': base,
  'omai-engine/numerics/linalg': linalg,
}

interface Example {
  readonly file: string
  readonly line: number
  readonly code: string[]
}

function sources(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name)
    if (statSync(path).isDirectory()) return sources(path)
    return name.endsWith('.ts') && !name.includes('.test') && !name.startsWith('_') && name !== 'index.ts' ? [path] : []
  })
}

function examples(path: string): Example[] {
  const text = readFileSync(path, 'utf8').split('\n')
  const out: Example[] = []
  for (let i = 0; i < text.length; i++) {
    if (!/^\s*\*\s*```ts\s*$/.test(text[i] ?? '')) continue
    const code: string[] = []
    const start = i
    for (i++; i < text.length && !/^\s*\*\s*```\s*$/.test(text[i] ?? ''); i++) {
      code.push((text[i] ?? '').replace(/^\s*\*\s?/, ''))
    }
    out.push({ file: relative(import.meta.dirname, path), line: start + 1, code })
  }
  return out
}

function compileExample(code: readonly string[]): string {
  const body: string[] = []
  for (const line of code) {
    const imp = /^import\s+\{([^}]*)\}\s+from\s+'([^']+)'/.exec(line)
    if (imp !== null) {
      const names = (imp[1] ?? '')
        .split(',')
        .map((s) => s.trim())
        .filter((s) => s.length > 0 && !s.startsWith('type '))
      body.push(`const {${names.join(', ')}} = __mods[${JSON.stringify(imp[2])}];`)
      continue
    }
    const check = /^(.*?)\s*\/\/ => (.*)$/.exec(line)
    if (check !== null) {
      const expected = check[2] ?? ''
      if (expected === 'trace' || expected === 'figure') body.push(`;(${check[1]});`)
      else if (expected.startsWith('throws')) {
        body.push(
          `__throws(() => (${check[1]}), ${JSON.stringify(expected.slice(6).trim())}, ${JSON.stringify(line)});`,
        )
      } else body.push(`__check(() => (${check[1]}), () => (${expected}), ${JSON.stringify(line)});`)
      continue
    }
    body.push(line)
  }
  return stripTypeScriptTypes(body.join('\n'))
}

function close(actual: unknown, expected: unknown, path: string): string | undefined {
  if (typeof expected === 'number' && typeof actual === 'number') {
    if (Object.is(actual, expected) || (Number.isNaN(actual) && Number.isNaN(expected))) return undefined
    return Math.abs(actual - expected) <= 1e-12 + 1e-9 * Math.abs(expected)
      ? undefined
      : `${path}: ${actual} != ${expected}`
  }
  if (Array.isArray(expected) || ArrayBuffer.isView(expected)) {
    const a = actual as ArrayLike<unknown>
    const e = expected as ArrayLike<unknown>
    if (a === null || typeof a !== 'object' || a.length !== e.length) return `${path}: length differs`
    for (let i = 0; i < e.length; i++) {
      const r = close(a[i], e[i], `${path}[${i}]`)
      if (r !== undefined) return r
    }
    return undefined
  }
  if (expected !== null && typeof expected === 'object') {
    if (actual === null || typeof actual !== 'object') return `${path}: not an object`
    for (const [k, v] of Object.entries(expected)) {
      const r = close((actual as Record<string, unknown>)[k], v, `${path}.${k}`)
      if (r !== undefined) return r
    }
    return undefined
  }
  return Object.is(actual, expected) ? undefined : `${path}: ${String(actual)} != ${String(expected)}`
}

describe('documentation examples', () => {
  for (const path of sources(import.meta.dirname)) {
    for (const ex of examples(path)) {
      it(`${ex.file}:${ex.line}`, () => {
        const run = new Function('__mods', '__check', '__throws', compileExample(ex.code)) as (
          mods: typeof MODULES,
          check: (actual: () => unknown, expected: () => unknown, text: string) => void,
          throws: (f: () => unknown, code: string, text: string) => void,
        ) => void
        run(
          MODULES,
          (actual, expected, text) => {
            expect(close(actual(), expected(), 'value'), text).toBeUndefined()
          },
          (f, code, text) => {
            expect(f, text).toThrow()
            try {
              f()
            } catch (e) {
              if (code.length > 0) expect((e as { code?: string }).code, text).toBe(code)
            }
          },
        )
      })
    }
  }
})
