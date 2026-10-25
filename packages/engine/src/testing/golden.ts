// Loading golden fixtures written by the Python oracle (oracle/mdml_oracle/fixture.py). Node only; not published.
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { z } from 'zod'
import {
  type DType,
  isTensor,
  type Tensor,
  type TensorJson,
  tensorFromJson,
  toFlat,
  type Value,
} from '../lib/tensor/index.ts'

const FIXTURES = join(import.meta.dirname, '../../fixtures')

const NumberJson = z.union([z.number(), z.enum(['inf', '-inf', 'nan'])])
const TensorJsonSchema: z.ZodType<TensorJson> = z.object({
  dtype: z.enum(['float64', 'float32', 'int32', 'bool']),
  shape: z.array(z.number().int()),
  data: z.array(NumberJson),
})

/** One reference computation from the oracle. */
export interface GoldenCase {
  readonly name: string
  readonly fn: string
  readonly inputs: readonly unknown[]
  readonly params: Readonly<Record<string, unknown>>
  readonly output: unknown
  readonly rtol: number
  readonly atol: number
}

/** A fixture file. */
export interface Fixture {
  readonly schema: 'mdml.fixture/1'
  readonly source: string
  readonly reference: Readonly<Record<string, string>>
  readonly cases: readonly GoldenCase[]
}

const FixtureSchema: z.ZodType<Fixture> = z.object({
  schema: z.literal('mdml.fixture/1'),
  source: z.string(),
  reference: z.record(z.string(), z.string()),
  cases: z.array(
    z.object({
      name: z.string(),
      fn: z.string(),
      inputs: z.array(z.unknown()),
      params: z.record(z.string(), z.unknown()),
      output: z.unknown(),
      rtol: z.number(),
      atol: z.number(),
    }),
  ),
})

/** Load and validate a fixture by its path under packages/engine/fixtures (without `.json`). */
export function loadFixture(path: string): Fixture {
  return FixtureSchema.parse(JSON.parse(readFileSync(join(FIXTURES, `${path}.json`), 'utf8')))
}

/** Whether a JSON value is an encoded tensor. */
export function isTensorJson(x: unknown): x is TensorJson {
  return TensorJsonSchema.safeParse(x).success
}

/** Decode an encoded tensor (or pass other JSON through). */
export function decode(x: unknown): unknown {
  return isTensorJson(x) ? tensorFromJson(x) : x
}

/** Assert that a value matches an expected encoded tensor within the case's tolerances (numpy `allclose`, NaN == NaN). */
export function expectMatches(
  actual: Value | number,
  expected: unknown,
  rtol: number,
  atol: number,
  label: string,
): void {
  const want = TensorJsonSchema.parse(expected)
  const got =
    typeof actual === 'number'
      ? { shape: [] as number[], data: [actual], dtype: 'float64' as DType }
      : asConcrete(actual)
  if (got.shape.length !== want.shape.length || got.shape.some((n, i) => n !== want.shape[i])) {
    throw new Error(`${label}: shape [${got.shape.join(', ')}] differs from [${want.shape.join(', ')}]`)
  }
  const values = typeof actual === 'number' ? [actual] : toFlat(actual)
  const reference = tensorFromJson(want)
  values.forEach((g, i) => {
    const w = reference.data[i] ?? Number.NaN
    const ok = Object.is(g, w) || (Number.isNaN(g) && Number.isNaN(w)) || Math.abs(g - w) <= atol + rtol * Math.abs(w)
    if (!ok) throw new Error(`${label}: element ${i} is ${g}, expected ${w}`)
  })
}

function asConcrete(x: Value): Tensor {
  if (!isTensor(x)) throw new Error('expected a concrete tensor')
  return x
}
