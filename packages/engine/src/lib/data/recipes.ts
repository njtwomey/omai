/**
 * Recipes: a generator, its knobs, a list of modifiers and a seed, held as plain data so a dataset can be remade
 * exactly, described in one line, or shared in a link. `makeFromRecipe` derives every step's key from the seed, the zod
 * `RecipeSchema` validates recipes, and `encodeRecipe` and `parseRecipe` turn them into URL-safe strings and back.
 *
 * @index structure Recipes: a generator, knobs, modifiers and seed as plain data.
 */

import { z } from 'zod'
import { OmaiError } from '../base/index.ts'
import { type Key, keyFromPath } from '../random/index.ts'
import { contains, type DictSpace, defaults, type SpaceValue } from '../space/index.ts'
import type { DatasetGenerator, Generated, GeneratorKnobs } from './generator.ts'
import {
  correlatedFeaturesModifier,
  missingModifier,
  nuisanceFeaturesModifier,
  outliersModifier,
  transformModifier,
} from './modifiers/features.ts'
import {
  cutmixModifier,
  cutoutModifier,
  mixupModifier,
  randomCropModifier,
  randomFlipModifier,
} from './modifiers/images.ts'
import {
  classImbalanceModifier,
  labelNoiseModifier,
  labelShiftModifier,
  prevalenceModifier,
} from './modifiers/labels.ts'
import { duplicatesModifier, leakModifier } from './modifiers/leaks.ts'
import {
  conceptDriftModifier,
  covariateShiftModifier,
  shortcutModifier,
  subpopulationShiftModifier,
} from './modifiers/shift.ts'

/**
 * A named, keyed change to generated data with its knobs as a `Space` (label noise, missingness, shift), so recipes can
 * hold modifiers as data and controls can be generated for them.
 *
 * @concept datasets
 */
export interface Modifier {
  readonly name: string
  readonly description: string
  readonly knobs: DictSpace
  apply(key: Key, input: Generated, knobs: GeneratorKnobs): Generated
}

/** A generator plus modifiers with their knobs and the seed, as plain data: everything needed to remake a dataset. */
export interface Recipe {
  readonly generator: string
  readonly knobs?: GeneratorKnobs
  /** The seed every key derives from (default 0). */
  readonly seed?: number
  readonly modifiers?: readonly { readonly name: string; readonly knobs?: GeneratorKnobs }[]
}

/** Where {@link makeFromRecipe} finds generators and modifiers by name. */
export interface RecipeCatalogue {
  readonly generators: Readonly<Record<string, DatasetGenerator>>
  /** Default {@link builtinModifiers}. */
  readonly modifiers?: Readonly<Record<string, Modifier>>
}

/**
 * The modifiers of `omai-engine/data` by name.
 *
 * @example
 * ```ts
 * import { builtinModifiers } from 'omai-engine/data'
 * Object.keys(builtinModifiers).includes('missing') // => true
 * ```
 */
export const builtinModifiers: Readonly<Record<string, Modifier>> = Object.fromEntries(
  [
    labelNoiseModifier,
    prevalenceModifier,
    labelShiftModifier,
    classImbalanceModifier,
    missingModifier,
    outliersModifier,
    nuisanceFeaturesModifier,
    transformModifier,
    correlatedFeaturesModifier,
    covariateShiftModifier,
    conceptDriftModifier,
    randomCropModifier,
    randomFlipModifier,
    cutoutModifier,
    mixupModifier,
    cutmixModifier,
    leakModifier,
    duplicatesModifier,
    subpopulationShiftModifier,
    shortcutModifier,
  ].map((m) => [m.name, m]),
)

function resolveKnobs(space: DictSpace, given: GeneratorKnobs | undefined, owner: string): GeneratorKnobs {
  for (const [name, value] of Object.entries(given ?? {})) {
    const s = space.spaces[name]
    if (s === undefined) throw new OmaiError('value', `${owner}: unknown knob ${name}`)
    if (!contains(s, value))
      throw new OmaiError('value', `${owner}: knob ${name} = ${String(value)} is outside its space`)
  }
  return { ...(defaults(space) as GeneratorKnobs), ...(given ?? {}) }
}

/**
 * Make a dataset from a recipe: the generator runs with key `keyFromPath(seed, 'recipe', generator)`, then each
 * modifier `i` with `keyFromPath(seed, 'recipe', i, name)`, so a recipe always gives the same data. The data's
 * metadata records the recipe.
 *
 * @param recipe - The recipe.
 * @param catalogue - Generators (and modifiers) by name.
 * @returns The data and its truth.
 * @throws OmaiError with code `value` for an unknown generator, modifier or knob.
 * @index function Remake a dataset exactly from its recipe.
 * @example
 * ```ts
 * import { makeFromRecipe } from 'omai-engine/data'
 * import { friedman1 } from 'omai-engine/datasets'
 * const g = makeFromRecipe({ generator: 'friedman1', knobs: { n: 30 }, seed: 1, modifiers: [{ name: 'missing', knobs: { rate: 0.1 } }] }, { generators: { friedman1 } })
 * g.data.x.shape // => [30, 10]
 * ```
 */
export function makeFromRecipe(recipe: Recipe, catalogue: RecipeCatalogue): Generated {
  const generator = catalogue.generators[recipe.generator]
  if (generator === undefined) throw new OmaiError('value', `makeFromRecipe: unknown generator ${recipe.generator}`)
  const seed = recipe.seed ?? 0
  const knobs = resolveKnobs(generator.knobs, recipe.knobs, generator.name)
  let out = generator.generate(keyFromPath(seed, 'recipe', generator.name), knobs)
  const modifiers = catalogue.modifiers ?? builtinModifiers
  const stamps: { name: string; knobs: GeneratorKnobs }[] = []
  for (const [i, m] of (recipe.modifiers ?? []).entries()) {
    const modifier = modifiers[m.name]
    if (modifier === undefined) throw new OmaiError('value', `makeFromRecipe: unknown modifier ${m.name}`)
    const mk = resolveKnobs(modifier.knobs, m.knobs, modifier.name)
    out = modifier.apply(keyFromPath(seed, 'recipe', i, m.name), out, mk)
    stamps.push({ name: m.name, knobs: mk })
  }
  const meta = { ...out.data.meta, recipe: { generator: generator.name, knobs, seed, modifiers: stamps } }
  return { data: { ...out.data, meta } as Generated['data'], truth: out.truth }
}

function knobText(knobs: GeneratorKnobs | undefined): string {
  const entries = Object.entries(knobs ?? {})
  return entries.length === 0 ? '' : ` (${entries.map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(', ')})`
}

/**
 * A one-line human description of a recipe.
 *
 * @param recipe - The recipe.
 * @returns The description.
 * @example
 * ```ts
 * import { describeRecipe } from 'omai-engine/data'
 * describeRecipe({ generator: 'moons', knobs: { n: 200 }, seed: 3, modifiers: [{ name: 'labelNoise', knobs: { rate: 0.2 } }] }) // => 'moons (n=200), then labelNoise (rate=0.2); seed 3'
 * ```
 */
export function describeRecipe(recipe: Recipe): string {
  const steps = [
    `${recipe.generator}${knobText(recipe.knobs)}`,
    ...(recipe.modifiers ?? []).map((m) => `${m.name}${knobText(m.knobs)}`),
  ]
  return `${steps.join(', then ')}; seed ${recipe.seed ?? 0}`
}

const SpaceValueSchema: z.ZodType<SpaceValue> = z.lazy(() =>
  z.union([z.number(), z.string(), z.boolean(), z.array(SpaceValueSchema), z.record(z.string(), SpaceValueSchema)]),
) as z.ZodType<SpaceValue>

/**
 * Schema for a {@link Recipe}.
 *
 * @example
 * ```ts
 * import { RecipeSchema } from 'omai-engine/data'
 * RecipeSchema.safeParse({ generator: 'moons', seed: 1 }).success // => true
 * ```
 */
export const RecipeSchema: z.ZodType<Recipe> = z.object({
  generator: z.string(),
  knobs: z.record(z.string(), SpaceValueSchema).optional(),
  seed: z.number().int().nonnegative().optional(),
  modifiers: z
    .array(z.object({ name: z.string(), knobs: z.record(z.string(), SpaceValueSchema).optional() }))
    .optional(),
}) as z.ZodType<Recipe>

function toBase64Url(text: string): string {
  const bytes = new TextEncoder().encode(text)
  let binary = ''
  for (const b of bytes) binary += String.fromCharCode(b)
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}

function fromBase64Url(text: string): string {
  const b64 = text.replace(/-/g, '+').replace(/_/g, '/')
  const binary = atob(b64 + '='.repeat((4 - (b64.length % 4)) % 4))
  return new TextDecoder().decode(Uint8Array.from(binary, (c) => c.charCodeAt(0)))
}

/**
 * Encode a recipe as a compact URL-safe string (base64url of its JSON), for sharing a dataset in a link.
 *
 * @param recipe - The recipe.
 * @returns The encoded recipe.
 * @example
 * ```ts
 * import { encodeRecipe } from 'omai-engine/data'
 * encodeRecipe({ generator: 'moons' }) // => 'eyJnZW5lcmF0b3IiOiJtb29ucyJ9'
 * ```
 */
export function encodeRecipe(recipe: Recipe): string {
  return toBase64Url(JSON.stringify(RecipeSchema.parse(recipe)))
}

/**
 * Parse a recipe from its encoded form or from JSON, validating it.
 *
 * @param text - An {@link encodeRecipe} string or recipe JSON.
 * @returns The recipe.
 * @throws OmaiError with code `schema` when the text is not a valid recipe.
 * @example
 * ```ts
 * import { encodeRecipe, parseRecipe } from 'omai-engine/data'
 * parseRecipe(encodeRecipe({ generator: 'moons', seed: 2 })).seed // => 2
 * ```
 */
export function parseRecipe(text: string): Recipe {
  const trimmed = text.trim()
  let value: unknown
  try {
    value = JSON.parse(trimmed.startsWith('{') ? trimmed : fromBase64Url(trimmed))
  } catch {
    throw new OmaiError('schema', 'parseRecipe: not JSON or an encoded recipe')
  }
  const parsed = RecipeSchema.safeParse(value)
  if (!parsed.success) throw new OmaiError('schema', `parseRecipe: ${parsed.error.message}`)
  return parsed.data
}
