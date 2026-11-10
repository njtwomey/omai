/**
 * What a benchmark model is, and how one is registered: a `BenchmarkModel` is a model (a spec written with `defineModel`,
 * or a program where the spec layer cannot say it yet) with a title, a description, tags and references, a seeded data
 * generator beside its canonical data, the truth (a closed form, a quadrature or a documented reference, as marginal
 * oracles the validation kernel's `agree` reads), and the plans for the inference methods that apply to it: which one,
 * with what knobs and budget, whether it must agree with the truth or is only reported, and why. Nothing here registers
 * on import; `benchmarkEntries` and `benchmarkMethods` turn a list of models into what `modelRegistry` and the
 * producer's `methodRegistry` take, so a worker runs a benchmark by name (`nuts/eight-schools-centred`).
 */

import { type Json, OmaiError } from '../../base/index.ts'
import type { Key } from '../../random/index.ts'
import { key as makeKey } from '../../random/index.ts'
import {
  type AnyInferenceMethod,
  inferenceMethodFor,
  type MethodModel,
  type ModelEntry,
  type ModelRegistry,
  methodKinds,
  modelRegistry,
} from '../inference/index.ts'
import type { Verdict } from '../inference/method.ts'
import { analysisOf } from '../inference/model-input.ts'
import { type BindOptions, type BoundModel, type DataInput, type ModelDefinition, toProgram } from '../model/index.ts'
import type { Model } from '../primitives.ts'
import type { AgreementTolerances, MarginalOracle } from '../validation/index.ts'

/** The data of a run with what made it. */
export interface BenchmarkData {
  /** The values of the model's data nodes (nested numbers, so it is JSON). */
  readonly data: DataInput
  /** Options of `bind` that go with the data (sizes of dimensions no data fixes). */
  readonly options?: BindOptions
  /** The parameters that generated the data, by name: the generating truth, where the data were simulated. */
  readonly generating: Readonly<Record<string, Json>>
}

/** How a truth was obtained, from the most to the least exact. */
export type TruthKind =
  /** A closed form (conjugate pairs, linear-Gaussian filters). */
  | 'exact'
  /** Quadrature of the posterior density (`gridPosterior`), exact to the accuracy of the rule. */
  | 'quadrature'
  /** The density of the target is known and iid draws or analytic marginals come from it (funnel, banana, mixtures). */
  | 'analytic'
  /** A long reference run whose moments are stored (never draws); what the source says. */
  | 'reference'

/**
 * How much work a truth may take: `full` is the committed resolution (a quadrature at tens of points per axis can take tens
 * of seconds), `fast` a coarse one for unit tests (the same oracles, a few points per axis, accurate to about two digits).
 */
export type TruthQuality = 'full' | 'fast'

/** What the methods are compared with: the one-dimensional marginals of the posterior, and the evidence where known. */
export interface BenchmarkTruth {
  readonly kind: TruthKind
  /** Where it comes from, in a sentence (the formula, the quadrature and its resolution, the reference run). */
  readonly source: string
  /** log p(data), where known. */
  readonly logEvidence?: number
  /** The marginal oracle of each scalar quantity, by name (`mu`, `theta[3]`): the names `agree` and posteriors use. */
  readonly marginals: Readonly<Record<string, MarginalOracle>>
}

/**
 * One inference method to run on a benchmark model: its kind (a key of `methodKinds`), knobs (JSON, as a worker request
 * carries them), the number of steps to give the session, and what to expect. A plan with `expect: 'agree'` must be
 * within the tolerances of the truth; one with `expect: 'differ'` must be outside them somewhere (the documented failure:
 * a Laplace fit on the banana, a sampler stuck in one mode), so a method that starts to agree is a surprise; one with
 * `expect: 'report'` is run and tabulated only. The `note` says why, or how it is expected to deviate (a mean-field fit
 * too narrow, a Laplace fit missing skew), so a reader of the table knows what is a finding and what is the design.
 */
export interface MethodPlan {
  /** A key of `methodKinds` (`nuts`, `svi`, `laplace`, `importance`, `tempered-smc`, `gibbs`, ...). */
  readonly kind: string
  /** The knobs of the method (only numbers, flags and choices cross a worker's wire). */
  readonly config: Readonly<Record<string, Json>>
  /** The steps of the session (an upper bound for the methods that stop themselves). */
  readonly steps: number
  readonly expect: 'agree' | 'differ' | 'report'
  /** Why it is expected to agree, or how it is expected to differ. */
  readonly note: string
  /** The quantities to compare (default: those of the truth). */
  readonly quantities?: readonly string[]
  /** Tolerances of an `agree` plan (default those of `agreementWithin`). */
  readonly tolerances?: AgreementTolerances
}

/**
 * A benchmark model: the model, its data, its truth and its plans. Immutable; the truth is computed when asked (a
 * quadrature can take seconds).
 *
 * @concept model-spec
 * @index structure A benchmark model: a spec with a data generator, the truth, references, tags and method plans.
 */
export interface BenchmarkModel {
  /** Registry name (`eight-schools-centred`): lower case with dashes, unique in a suite. */
  readonly name: string
  readonly title: string
  /** One or two sentences: what the model is and why it is a benchmark. */
  readonly description: string
  /** Free tags for a selector (`hierarchical`, `funnel`, `conjugate`, `glm`, `multimodal`, `state-space`, ...). */
  readonly tags: readonly string[]
  /** Where the model and its data come from. */
  readonly references: readonly string[]
  /**
   * `posterior`: a model with observed nodes, whose posterior is wanted. `density`: a target with no data (the funnel, the
   * banana): the spec's capability checks read an observed node as the likelihood and refuse a spec without one, so the
   * methods are made on the compiled program, whose checks are the engines' own.
   */
  readonly target: 'posterior' | 'density'
  /** The spec, when the model is one (`undefined` for a model written as a program). */
  readonly definition: ModelDefinition | undefined
  /** The canonical data: an embedded dataset, or the generator at seed 0. */
  readonly data: BenchmarkData
  /** Data from the generator at a key; `size` scales the number of rows, where the model has them. */
  generate(key: Key, size?: number): BenchmarkData
  /** The bound spec (throws for a model written as a program). */
  bound(data?: BenchmarkData): BoundModel
  /** The program every engine runs: the compiled spec, or the program itself. */
  program(data?: BenchmarkData): Model
  /** What a method is made for: the bound spec of a `posterior` model, the program otherwise. */
  methodModel(data?: BenchmarkData): MethodModel
  /**
   * Quantities derived from the draws of a method, by name, beyond the sites themselves: a label-switching-aware summary
   * (the sorted component means of a mixture, which a relabelling of the draws leaves unchanged). The truth has oracles
   * for them under the same names. Absent when the model has none.
   */
  summarise?(draws: Readonly<Record<string, ArrayLike<number>>>): Record<string, ArrayLike<number>>
  /** The truth for given data (default the canonical data) at a quality (default `full`). */
  truth(data?: BenchmarkData, quality?: TruthQuality): BenchmarkTruth
  /** The method plans, with budgets multiplied by `scale` (1 is the committed budget). */
  plans(scale?: number): readonly MethodPlan[]
}

/** The parts of a benchmark model; {@link defineBenchmark} adds the accessors. */
export interface BenchmarkParts {
  readonly name: string
  readonly title: string
  readonly description: string
  readonly tags: readonly string[]
  readonly references: readonly string[]
  readonly target?: 'posterior' | 'density'
  /** The spec (give this, or `program`). */
  readonly definition?: ModelDefinition
  /** A model written as a program, closed over its data (give this, or `definition`). */
  readonly program?: (data: BenchmarkData) => Model
  /** The generator: seeded by the key, `size` scales the rows. */
  readonly generate: (key: Key, size?: number) => BenchmarkData
  /** An embedded dataset that is the canonical data (default: the generator at seed 0). */
  readonly canonical?: BenchmarkData
  readonly truth: (data: BenchmarkData, quality: TruthQuality) => BenchmarkTruth
  /** Derived quantities of the draws (see {@link BenchmarkModel.summarise}). */
  readonly summarise?: (draws: Readonly<Record<string, ArrayLike<number>>>) => Record<string, ArrayLike<number>>
  readonly plans: (scale: number) => readonly MethodPlan[]
}

/**
 * Make a benchmark model from its parts: checks the name and that there is a spec or a program, and adds the accessors
 * (`bound`, `program`, `methodModel`) that bind the canonical data unless other data is given.
 *
 * @param parts - The model, generator, truth and plans.
 * @returns The benchmark model.
 * @throws OmaiError with code `value` for a name that is not lower case with dashes, no spec and no program, or both.
 * @concept model-spec
 * @index function Make a benchmark model: a spec or program with generator, truth, tags, references and method plans.
 * @example
 * ```ts
 * import { betaBernoulli } from 'omai-engine/probprog/models'
 * [betaBernoulli.name, betaBernoulli.bound().sizes, betaBernoulli.truth().kind] // => ['beta-bernoulli', { flip: 20 }, 'exact']
 * ```
 */
export function defineBenchmark(parts: BenchmarkParts): BenchmarkModel {
  if (!/^[a-z][a-z0-9]*(-[a-z0-9]+)*$/.test(parts.name))
    throw new OmaiError('value', `a benchmark name is lower case with dashes, got '${parts.name}'`)
  if ((parts.definition === undefined) === (parts.program === undefined))
    throw new OmaiError(
      'value',
      `benchmark '${parts.name}': give a spec (definition) or a program, not both or neither`,
    )
  let canonical: BenchmarkData | undefined = parts.canonical
  const dataOf = (data: BenchmarkData | undefined): BenchmarkData => {
    if (data !== undefined) return data
    canonical ??= parts.generate(makeKey(0))
    return canonical
  }
  const bound = (data?: BenchmarkData): BoundModel => {
    if (parts.definition === undefined)
      throw new OmaiError('value', `benchmark '${parts.name}' is a program, not a spec: use program()`)
    const d = dataOf(data)
    return parts.definition.bind(d.data, d.options)
  }
  const program = (data?: BenchmarkData): Model => {
    const d = dataOf(data)
    return parts.program !== undefined ? parts.program(d) : toProgram(bound(d))
  }
  const target = parts.target ?? 'posterior'
  return {
    name: parts.name,
    title: parts.title,
    description: parts.description,
    tags: parts.tags,
    references: parts.references,
    target,
    definition: parts.definition,
    get data(): BenchmarkData {
      return dataOf(undefined)
    },
    generate: parts.generate,
    bound,
    program,
    methodModel: (data) => (parts.definition !== undefined && target === 'posterior' ? bound(data) : program(data)),
    truth: (data, quality = 'full') => parts.truth(dataOf(data), quality),
    ...(parts.summarise === undefined ? {} : { summarise: parts.summarise }),
    plans: (scale = 1) => parts.plans(scale),
  }
}

/**
 * The spec models of a suite as the entries `modelRegistry` takes, each with its canonical data, so a worker's module
 * registers them by name.
 *
 * @param models - The benchmark models (a model written as a program has no entry: a registry holds specs).
 * @returns The entries by model name.
 * @concept model-spec
 * @example
 * ```ts
 * import { benchmarkEntries, betaBernoulli, normalNormal } from 'omai-engine/probprog/models'
 * Object.keys(benchmarkEntries([betaBernoulli, normalNormal])) // => ['beta-bernoulli', 'normal-normal']
 * ```
 */
export function benchmarkEntries(models: readonly BenchmarkModel[]): Record<string, ModelEntry> {
  const out: Record<string, ModelEntry> = {}
  for (const m of models) {
    if (m.definition === undefined) continue
    out[m.name] = {
      definition: m.definition,
      data: m.data.data,
      ...(m.data.options === undefined ? {} : { options: m.data.options }),
    }
  }
  return out
}

/**
 * A `ModelRegistry` of the spec models of a suite.
 *
 * @param models - The benchmark models.
 * @returns The registry (`need(name)` is the model bound to its canonical data, `bind(name, data)` to other data).
 * @concept model-spec
 * @example
 * ```ts
 * import { benchmarkRegistry, betaBernoulli } from 'omai-engine/probprog/models'
 * benchmarkRegistry([betaBernoulli]).need('beta-bernoulli').sizes // => { flip: 20 }
 * ```
 */
export function benchmarkRegistry(models: readonly BenchmarkModel[]): ModelRegistry {
  return modelRegistry(benchmarkEntries(models))
}

/**
 * The inference methods on offer for a suite, named `<kind>/<model>`, for the producer's `methodRegistry`: a spec model
 * of a `posterior` target goes through the registry's `inferenceMethods`; a `density` target or a program is made on its
 * program under the same naming.
 *
 * @param models - The benchmark models.
 * @param kinds - The kinds on offer (keys of `methodKinds`; default every kind).
 * @returns The methods.
 * @throws OmaiError with code `value` for a kind that is not in the table.
 * @concept inference-engines
 * @example
 * ```ts
 * import { benchmarkMethods, betaBernoulli } from 'omai-engine/probprog/models'
 * benchmarkMethods([betaBernoulli], ['conjugate', 'nuts']).map((m) => m.name) // => ['conjugate/beta-bernoulli', 'nuts/beta-bernoulli']
 * ```
 */
export function benchmarkMethods(
  models: readonly BenchmarkModel[],
  kinds: readonly string[] = Object.keys(methodKinds),
): AnyInferenceMethod[] {
  const registry = benchmarkRegistry(models)
  const out: AnyInferenceMethod[] = []
  for (const m of models)
    for (const kind of kinds) {
      if (m.definition !== undefined && m.target === 'posterior') out.push(inferenceMethodFor(registry, kind, m.name))
      else {
        const make = methodKinds[kind]
        if (make === undefined)
          throw new OmaiError(
            'value',
            `no inference method of kind '${kind}' (known: ${Object.keys(methodKinds).join(', ')})`,
          )
        out.push(make(m.program(), { name: `${kind}/${m.name}` }))
      }
    }
  return out
}

/** Whether a method kind applies to a benchmark model, and why. */
export interface MethodVerdict {
  readonly kind: string
  readonly verdict: Verdict
}

/**
 * The verdict of every method kind's `check` on a model (on the model a method would be made for: the bound spec, or the
 * program). The list of what applies is what a page greys methods out by; nothing here selects one.
 *
 * @param model - The benchmark model.
 * @param kinds - The kinds to ask (default every kind).
 * @returns One verdict per kind, in the order of `kinds`.
 * @concept inference-engines
 * @example
 * ```ts
 * import { betaBernoulli, methodVerdicts } from 'omai-engine/probprog/models'
 * methodVerdicts(betaBernoulli, ['conjugate', 'enumeration']).map((v) => [v.kind, v.verdict.applies]) // => [['conjugate', true], ['enumeration', false]]
 * ```
 */
export function methodVerdicts(
  model: BenchmarkModel,
  kinds: readonly string[] = Object.keys(methodKinds),
): MethodVerdict[] {
  const input = model.methodModel()
  const analysis = analysisOf(input)
  return kinds.map((kind) => {
    const make = methodKinds[kind]
    if (make === undefined) throw new OmaiError('value', `no inference method of kind '${kind}'`)
    const method = make(input, {})
    const verdict = method.check?.(analysis) ?? { applies: true, reason: 'applies to any model', missing: [] }
    return { kind, verdict }
  })
}
