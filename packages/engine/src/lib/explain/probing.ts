/**
 * Probing classifiers with control tasks (Alain and Bengio 2016, Hewitt and Liang 2019): a linear classifier trained
 * on a layer's activations to read off a property of the input, scored on held-out rows. A probe that reads a property
 * may be reading it from the layer, or learning it itself, so each probe is paired with a control task, the same
 * inputs with labels assigned at random per word type, which only a probe that memorises can fit; the difference of
 * the two accuracies is the probe's selectivity. The probe is a multinomial logistic regression of `linear`, so its
 * solver is that model's solver slot, and `probeLayers` runs it down a stack of layers for the where-is-it-readable map.
 */
import { OmaiError } from '../base/index.ts'
import { matrixView } from '../learn/validation.ts'
import { multinomial } from '../linear/index.ts'
import { foldIn, type Key, permutation, uniform } from '../random/index.ts'
import type { SpaceValue } from '../space/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'

/** Options of {@link probeAccuracy}. */
export interface ProbeOptions {
  /** The key of the train/test split (and the control task). */
  readonly key: Key
  /** The share of rows held out for testing (default 0.3). */
  readonly testFraction?: number
  /** The ridge penalty of the probe (default 0.01). */
  readonly alpha?: number
  /** How the probe's likelihood is minimised: a value of `smoothSolvers.space` (default Newton). */
  readonly solver?: SpaceValue
}

/** A probe's accuracies. */
export interface ProbeResult {
  readonly trainAccuracy: number
  readonly testAccuracy: number
  /** The accuracy of always predicting the commonest training class on the test rows. */
  readonly baseline: number
  readonly classes: number
}

function take(m: { readonly data: Float64Array; readonly d: number }, rows: readonly number[]): Tensor {
  const out = new Float64Array(rows.length * m.d)
  for (const [i, r] of rows.entries()) out.set(m.data.subarray(r * m.d, (r + 1) * m.d), i * m.d)
  return fromStorage(out, [rows.length, m.d], 'float64')
}

/**
 * Train a linear probe on activations and score it on held-out rows.
 *
 * @param activations - Activations `[n, d]`.
 * @param labels - One class label per row.
 * @param options - The key, the held-out share, the ridge penalty and the solver.
 * @returns Train and test accuracy, and the majority-class baseline.
 * @throws OmaiError with code `shape` if the labels do not match the rows.
 * @concept probing
 * @example
 * ```ts
 * import { probeAccuracy } from 'omai-engine/explain'
 * import { key } from 'omai-engine/random'
 * const x = Array.from({ length: 40 }, (_, i) => [i % 2, ((i * 7) % 5) / 5])
 * probeAccuracy(x, x.map((r) => r[0] as number), { key: key(0) }).testAccuracy // => 1
 * ```
 */
export function probeAccuracy(activations: TensorLike, labels: readonly number[], options: ProbeOptions): ProbeResult {
  const m = matrixView(activations, 'probeAccuracy')
  if (labels.length !== m.n) throw new OmaiError('shape', 'probeAccuracy: one label per row')
  const order = Array.from(permutation(options.key, m.n).data)
  const nTest = Math.max(1, Math.round(m.n * (options.testFraction ?? 0.3)))
  const test = order.slice(0, nTest)
  const train = order.slice(nTest)
  const classes = [...new Set(labels)].sort((a, b) => a - b)
  const yTrain = train.map((i) => labels[i] as number)
  const yTest = test.map((i) => labels[i] as number)
  const count = new Map<number, number>()
  for (const y of yTrain) count.set(y, (count.get(y) ?? 0) + 1)
  const majority = [...count.entries()].sort((a, b) => b[1] - a[1] || a[0] - b[0])[0]?.[0] ?? 0
  const baseline = yTest.filter((y) => y === majority).length / yTest.length
  if (new Set(yTrain).size < 2) {
    return { trainAccuracy: 1, testAccuracy: baseline, baseline, classes: classes.length }
  }
  const fitted = multinomial({
    alpha: options.alpha ?? 0.01,
    ...(options.solver === undefined ? {} : { solver: options.solver }),
  }).fit({ x: take(m, train), y: yTrain }, { key: options.key })
  const accuracy = (rows: readonly number[], y: readonly number[]): number => {
    const pred = toFlat(fitted.predict(take(m, rows)))
    return y.reduce((a, v, i) => a + (pred[i] === v ? 1 : 0), 0) / y.length
  }
  return {
    trainAccuracy: accuracy(train, yTrain),
    testAccuracy: accuracy(test, yTest),
    baseline,
    classes: classes.length,
  }
}

/**
 * A control task (Hewitt and Liang): every word type gets a label drawn at random from the empirical label
 * distribution and keeps it everywhere it occurs, so the labels carry the type's identity and nothing about the
 * property. A probe can only fit it by memorising the types.
 *
 * @param types - The type (word identity) of each row.
 * @param labels - The real labels, whose frequencies the control labels follow.
 * @param key - The key.
 * @returns Control labels, one per row.
 * @example
 * ```ts
 * import { controlTask } from 'omai-engine/explain'
 * import { key } from 'omai-engine/random'
 * const c = controlTask([5, 6, 5, 6], [0, 1, 0, 1], key(2))
 * c[0] === c[2] && c[1] === c[3] // => true
 * ```
 */
export function controlTask(types: readonly number[], labels: readonly number[], key: Key): number[] {
  if (types.length !== labels.length) throw new OmaiError('shape', 'controlTask: one type per label')
  const distinct = [...new Set(types)]
  const u = uniform(foldIn(key, 0), [distinct.length]).data
  const assigned = new Map<number, number>()
  distinct.forEach((t, i) => {
    const pick = labels[Math.min(labels.length - 1, Math.floor((u[i] as number) * labels.length))] as number
    assigned.set(t, pick)
  })
  return types.map((t) => assigned.get(t) as number)
}

/** A probe, its control and the selectivity. */
export interface SelectivityResult {
  readonly task: ProbeResult
  readonly control: ProbeResult
  /** Task accuracy minus control accuracy on the held-out rows: how much of the probe's success is the layer's. */
  readonly selectivity: number
}

/**
 * Probe a property and its control task on the same activations: the probe's test accuracy on the property, on the
 * control labels, and their difference (the selectivity). The split is the same for both.
 *
 * @param activations - Activations `[n, d]`.
 * @param labels - The property's labels.
 * @param types - The type of each row (a word identity), which the control labels follow.
 * @param options - The key and the probe's options.
 * @returns The two probes and the selectivity.
 * @concept probing
 * @example
 * ```ts
 * import { probeSelectivity } from 'omai-engine/explain'
 * import { key } from 'omai-engine/random'
 * const types = Array.from({ length: 60 }, (_, i) => i % 6)
 * const x = types.map((t) => [t % 2, (t * 3) % 5])
 * probeSelectivity(x, types.map((t) => t % 2), types, { key: key(1) }).task.testAccuracy // => 1
 * ```
 */
export function probeSelectivity(
  activations: TensorLike,
  labels: readonly number[],
  types: readonly number[],
  options: ProbeOptions,
): SelectivityResult {
  const task = probeAccuracy(activations, labels, options)
  const control = probeAccuracy(activations, controlTask(types, labels, foldIn(options.key, 7)), options)
  return { task, control, selectivity: task.testAccuracy - control.testAccuracy }
}

/** The probe at each layer. */
export interface LayerProbes {
  /** Test accuracy of the property probe at each layer. */
  readonly accuracy: readonly number[]
  /** Test accuracy of the control probe at each layer, when types were given. */
  readonly control?: readonly number[]
  /** The selectivity at each layer, when types were given. */
  readonly selectivity?: readonly number[]
  readonly baseline: number
}

/**
 * Probe every layer of a stack for a property: the accuracy by depth, with the control task and selectivity when the
 * rows' types are given. Where the accuracy climbs above the baseline is where the property becomes linearly readable.
 *
 * @param layers - Activations per layer, each `[n, d]` (the widths may differ).
 * @param labels - The property's labels.
 * @param options - The probe's options, plus the row types for controls.
 * @returns The accuracy by layer.
 * @concept probing
 * @example
 * ```ts
 * import { probeLayers } from 'omai-engine/explain'
 * import { key } from 'omai-engine/random'
 * const y = Array.from({ length: 40 }, (_, i) => i % 2)
 * const noise = y.map((_, i) => [((i * 37) % 11) / 11])
 * const signal = y.map((v, i) => [v, noise[i]?.[0] ?? 0])
 * probeLayers([noise, signal], y, { key: key(0) }).accuracy[1] // => 1
 * ```
 */
export function probeLayers(
  layers: readonly TensorLike[],
  labels: readonly number[],
  options: ProbeOptions & { readonly types?: readonly number[] },
): LayerProbes {
  const results = layers.map((l) =>
    options.types === undefined
      ? probeAccuracy(l, labels, options)
      : probeSelectivity(l, labels, options.types, options),
  )
  const task = results.map((r) => ('task' in r ? r.task : r))
  const out = {
    accuracy: task.map((r) => r.testAccuracy),
    baseline: task[0]?.baseline ?? 0,
  }
  if (options.types === undefined) return out
  const sel = results as SelectivityResult[]
  return { ...out, control: sel.map((r) => r.control.testAccuracy), selectivity: sel.map((r) => r.selectivity) }
}
