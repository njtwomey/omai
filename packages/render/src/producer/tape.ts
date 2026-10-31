/**
 * The tape as a file: `tapeOf` writes the sessions of a page (the recorded events of each variant and the probes the
 * page's panels read, at a spread of steps) as one JSON document, `parseTape` validates one, and `ReplaySession` is a
 * session read back from it: the same tape, the same probes, nothing recomputed, so a page can be reproduced from the
 * file. Probes are stored at the steps exported; between them a replayed session answers with the nearest stored one.
 */

import { type Json, MdmlError } from 'mdml-engine/base'
import {
  type Algorithm,
  type Config,
  type Contract,
  type Derived,
  type Method,
  parseRun,
  type Run,
  type RunStatus,
  type Session,
  type Slice,
  stateFromJson,
  stateToJson,
} from 'mdml-engine/trace'
import type { MethodRegistry } from './registry.ts'
import { canProbeAsync, contractOfKinds } from './remote.ts'

/** One probe stored in a tape file: a derived path at a step, with its arguments, and its value (or why it failed). */
export interface TapeProbe {
  readonly t: number
  readonly path: string
  readonly args?: Json
  readonly value?: Json
  readonly error?: string
}

/** One variant of a tape file. */
export interface TapeVariant {
  readonly label: string
  /** The configuration the variant ran with (defaults included). */
  readonly config: Json
  /** The step reached and how the run ended (`null` while it had not). */
  readonly head: number
  readonly status: RunStatus | null
  /** The recorded events. */
  readonly run: Run
  /** The kind of each path of the contract, so a replayed session knows which paths are probes. */
  readonly kinds: Readonly<Record<string, string>>
  /** The steps at which each event path fired. */
  readonly fired: Readonly<Record<string, readonly number[]>>
  readonly probes: readonly TapeProbe[]
}

/** The tape of a page as a file. */
export interface TapeFile {
  readonly schema: 'mdml.tape/1'
  /** The method's name in a registry, so the page can be rebuilt without running it. */
  readonly method: string
  readonly seed: number
  readonly title?: string
  /** The step the page was showing when it was exported. */
  readonly at?: number
  /** The page's layout, when the exporter had one. */
  readonly layout?: Json
  readonly variants: readonly TapeVariant[]
}

/** A derived path (and its arguments) a page reads, so a tape file can hold it. */
export interface ProbeRequest {
  readonly path: string
  readonly args?: Json
}

/** A session to put in a tape file, with its label. */
export interface TapedSession {
  readonly label: string
  // biome-ignore lint/suspicious/noExplicitAny: sessions of any method
  readonly session: Session<any, any>
}

/** What goes into a tape file besides the events. */
export interface TapeOptions {
  /** The derived paths to store, at each of the steps. */
  readonly requests: readonly ProbeRequest[]
  /** The most steps to store probes at (default 120): evenly spread, always with the first, the last and `at`. */
  readonly maxSteps?: number
  readonly at?: number
  readonly title?: string
  readonly layout?: Json
}

/**
 * The steps to store probes at: all of them when few, else an even spread that always holds the first step, the last
 * and the step being viewed.
 *
 * @param head - The last step of the tape.
 * @param max - The most steps.
 * @param at - A step to be sure of (the viewed one).
 * @returns The steps, ascending.
 */
export function probeSteps(head: number, max: number, at?: number): number[] {
  if (head < 0) return []
  if (head + 1 <= max) return Array.from({ length: head + 1 }, (_, i) => i)
  const out = new Set<number>([0, head])
  if (at !== undefined && at >= 0 && at <= head) out.add(Math.round(at))
  const n = Math.max(2, max - out.size)
  for (let i = 0; i < n; i++) out.add(Math.round((i * head) / (n - 1)))
  return [...out].sort((a, b) => a - b)
}

const keyOf = (path: string, args: Json | undefined): string => `${path}|${JSON.stringify(args ?? null)}`

/**
 * Write the sessions of a page as a tape file: each variant's events and the probes the requests name, at a spread of
 * steps (a remote session is asked in its worker; the answers are collected a few at a time).
 *
 * @param method - The method's registry name.
 * @param sessions - The sessions, with their labels.
 * @param options - The probes to store and the page's title, layout and step.
 * @returns The file, ready for `JSON.stringify`.
 */
export async function tapeOf(
  method: string,
  sessions: readonly TapedSession[],
  options: TapeOptions,
): Promise<TapeFile> {
  const variants: TapeVariant[] = []
  for (const { label, session } of sessions) {
    const head = session.t
    const steps = probeSteps(head, options.maxSteps ?? 120, options.at)
    const kinds = Object.fromEntries(Object.entries(session.contract).map(([path, source]) => [path, source.kind]))
    // Only derived paths can be probed; a request for another kind is a tape path and is already in the events.
    const wanted = options.requests.filter((r) => session.contract[r.path]?.kind === 'derived')
    const probes: TapeProbe[] = []
    const jobs = steps.flatMap((t) => wanted.map((r) => ({ t, r })))
    const one = async ({ t, r }: { t: number; r: ProbeRequest }): Promise<TapeProbe> => {
      const base = { t, path: r.path, ...(r.args === undefined ? {} : { args: r.args }) }
      try {
        const got: Derived = canProbeAsync(session)
          ? await session.probeAsync(t, r.path, r.args)
          : session.probe(t, r.path as never, r.args)
        return { ...base, value: stateToJson(got) }
      } catch (e) {
        return { ...base, error: e instanceof Error ? e.message : String(e) }
      }
    }
    for (let i = 0; i < jobs.length; i += 16) probes.push(...(await Promise.all(jobs.slice(i, i + 16).map(one))))
    const fired: Record<string, readonly number[]> = {}
    for (const [path, source] of Object.entries(session.contract))
      if (source.kind === 'event') fired[path] = session.stepsOf(path as never)
    variants.push({
      label,
      config: session.config as unknown as Json,
      head,
      status: session.status ?? null,
      run: session.run(),
      kinds,
      fired,
      probes,
    })
  }
  return {
    schema: 'mdml.tape/1',
    method,
    seed: sessions[0]?.session.seed ?? 0,
    ...(options.title === undefined ? {} : { title: options.title }),
    ...(options.at === undefined ? {} : { at: options.at }),
    ...(options.layout === undefined ? {} : { layout: options.layout }),
    variants,
  }
}

const isObject = (x: unknown): x is Record<string, unknown> => typeof x === 'object' && x !== null && !Array.isArray(x)

/**
 * Parse and validate a tape file from JSON text or a parsed value.
 *
 * @param input - The text or value.
 * @returns The file.
 * @throws MdmlError with code `schema` when it is not a tape file (or its events are not a valid run).
 */
export function parseTape(input: unknown): TapeFile {
  let value: unknown
  try {
    value = typeof input === 'string' ? JSON.parse(input) : input
  } catch {
    throw new MdmlError('schema', 'a tape file is JSON: this is not')
  }
  if (!isObject(value) || value['schema'] !== 'mdml.tape/1')
    throw new MdmlError('schema', "not a tape file (its schema should be 'mdml.tape/1')")
  if (typeof value['method'] !== 'string' || typeof value['seed'] !== 'number' || !Array.isArray(value['variants']))
    throw new MdmlError('schema', 'a tape file names a method and a seed and holds its variants')
  const variants = (value['variants'] as unknown[]).map((v, i): TapeVariant => {
    if (!isObject(v) || typeof v['label'] !== 'string' || typeof v['head'] !== 'number' || !isObject(v['kinds']))
      throw new MdmlError('schema', `variant ${i} of the tape file is malformed`)
    return {
      label: v['label'],
      config: (v['config'] ?? {}) as Json,
      head: v['head'],
      status: (v['status'] ?? null) as RunStatus | null,
      run: parseRun(v['run']),
      kinds: v['kinds'] as Record<string, string>,
      fired: (isObject(v['fired']) ? v['fired'] : {}) as Record<string, number[]>,
      probes: (Array.isArray(v['probes']) ? v['probes'] : []) as TapeProbe[],
    }
  })
  return {
    schema: 'mdml.tape/1',
    method: value['method'],
    seed: value['seed'],
    ...(typeof value['title'] === 'string' ? { title: value['title'] } : {}),
    ...(typeof value['at'] === 'number' ? { at: value['at'] } : {}),
    ...(value['layout'] === undefined ? {} : { layout: value['layout'] as Json }),
    variants,
  }
}

const unavailable = (what: string): never => {
  throw new MdmlError('value', `${what} is not available on a session replayed from a tape file`)
}

/**
 * A session read back from a tape file: it stands at the head the file recorded, its tape is the file's events and its
 * probes are the stored ones (the nearest stored step when the asked one was not stored). Nothing is computed.
 *
 * @concept log-contract
 */
export class ReplaySession<S, C extends Config = Config, K extends string = string> implements Session<S, C, K> {
  readonly method: Method<S, C, K>
  readonly config: C
  readonly seed: number
  readonly contract: Contract<S, K>
  private readonly tape: TapeVariant
  private readonly byPath = new Map<string, { steps: number[]; at: Map<number, TapeProbe> }>()
  private readonly decoded = new Map<TapeProbe, Derived>()

  constructor(method: Method<S, C, K>, tape: TapeVariant, seed: number) {
    this.method = method
    this.tape = tape
    this.seed = seed
    this.config = tape.config as unknown as C
    this.contract = contractOfKinds(tape.kinds) as unknown as Contract<S, K>
    for (const p of tape.probes) {
      const k = keyOf(p.path, p.args)
      const entry = this.byPath.get(k) ?? { steps: [] as number[], at: new Map<number, TapeProbe>() }
      entry.steps.push(p.t)
      entry.at.set(p.t, p)
      this.byPath.set(k, entry)
    }
    for (const e of this.byPath.values()) e.steps.sort((a, b) => a - b)
  }

  get algorithm(): Algorithm<S, K> {
    return unavailable('the algorithm')
  }

  get state(): S {
    return unavailable('the state')
  }

  get t(): number {
    return this.tape.head
  }

  get status(): RunStatus | undefined {
    return this.tape.status ?? undefined
  }

  advance(): Slice<S> {
    return {
      state: undefined as S,
      t: this.tape.head,
      ...(this.tape.status === null ? {} : { status: this.tape.status }),
    }
  }

  stateAt(_t: number): S {
    return unavailable('stateAt')
  }

  fork(): Session<S, C, K> {
    return unavailable('fork')
  }

  probe(t: number, path: K, args?: Json): Derived {
    const entry = this.byPath.get(keyOf(path, args))
    if (entry === undefined || entry.steps.length === 0)
      throw new MdmlError('value', `the tape file holds no probe of '${path}'`)
    let step = entry.steps[0] as number
    for (const s of entry.steps) if (Math.abs(s - t) < Math.abs(step - t)) step = s
    const p = entry.at.get(step) as TapeProbe
    if (p.error !== undefined) throw new MdmlError('value', p.error)
    let value = this.decoded.get(p)
    if (value === undefined) {
      value = stateFromJson(p.value ?? null) as Derived
      this.decoded.set(p, value)
    }
    return value
  }

  project(name: string, options?: Json): Derived | undefined {
    const p = this.method.projections?.[name]
    if (p === undefined) throw new MdmlError('value', `${this.method.name}: no projection '${name}'`)
    return p(this.run(), options)
  }

  run(): Run {
    return this.tape.run
  }

  stepsOf(event: K): readonly number[] {
    return this.tape.fired[event] ?? []
  }
}

/**
 * The sessions of a tape file, replayed: the method is looked up by the file's name.
 *
 * @param file - The tape file.
 * @param registry - The methods on offer.
 * @returns A replayed session per variant, with its label.
 * @throws MdmlError with code `value` when the registry does not know the file's method.
 */
export function replaySessions(
  file: TapeFile,
  registry: MethodRegistry,
): readonly { readonly label: string; readonly session: ReplaySession<unknown> }[] {
  const method = registry.need(file.method)
  return file.variants.map((v) => ({ label: v.label, session: new ReplaySession(method, v, file.seed) }))
}
