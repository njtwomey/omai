// Private exposition helpers of the neural methods (methods/neural): the loss and score curves of a training run,
// and a hidden layer's representation of the training rows at each snapshot, projected to two dimensions.
import {
  type EmbeddingSteps,
  type Exposition,
  embeddingSteps,
  type LearningCurve,
  learningCurve,
} from '../../lib/exposition/index.ts'
import { eigh } from '../../lib/numerics/linalg/index.ts'
import { type Tensor, toFlat } from '../../lib/tensor/index.ts'
import type { EpochReport, Snapshot, Training } from './_train.ts'

/**
 * The loss per epoch: the mean minibatch loss (scikit-learn's loss_curve_) against the validation loss, or, without
 * validation rows, against the training loss at the end of each epoch.
 */
export function lossCurve(training: Training): LearningCurve {
  const validation = training.validationLoss ?? training.evalLoss
  return learningCurve({
    sizes: training.epochs,
    train: training.trainLoss,
    validation,
    resource: 'epoch',
    labels: {
      train: 'training loss (mean over the epoch)',
      validation: training.validationLoss !== undefined ? 'validation loss' : 'training loss (end of the epoch)',
    },
  })
}

/** The score per epoch on the training rows and, when held out, the validation rows (else the training score again). */
export function scoreCurve(training: Training, name: string): LearningCurve | undefined {
  const train = training.trainScore
  if (train === undefined) return undefined
  const validation = training.validationScore
  return learningCurve({
    sizes: training.epochs,
    train,
    validation: validation ?? train,
    resource: 'epoch',
    labels: {
      train: `training ${name}`,
      validation: validation === undefined ? `training ${name}` : `validation ${name}`,
    },
  })
}

/** The minibatch loss of every step. */
export function stepLossCurve(training: Training): LearningCurve {
  const sizes = training.stepLoss.map((_, i) => i + 1)
  return learningCurve({
    sizes,
    train: training.stepLoss,
    validation: training.stepLoss,
    resource: 'step',
    labels: { train: 'minibatch loss', validation: 'minibatch loss' },
  })
}

/** Rows of a matrix tensor `[n, h]` (or `[n]`) as arrays. */
export function rowsOf(t: Tensor): number[][] {
  const n = t.shape[0] ?? 0
  const flat = toFlat(t)
  const h = n === 0 ? 0 : flat.length / n
  return Array.from({ length: n }, (_, i) => flat.slice(i * h, (i + 1) * h))
}

// The top two principal axes of centred rows (a covariance eigendecomposition: hidden layers are narrow).
function principalPlane(rows: readonly (readonly number[])[]): { mean: number[]; axes: number[][] } {
  const n = rows.length
  const h = rows[0]?.length ?? 0
  const mean = Array.from({ length: h }, (_, j) => rows.reduce((s, r) => s + (r[j] ?? 0), 0) / Math.max(1, n))
  if (h <= 2)
    return { mean, axes: Array.from({ length: h }, (_, j) => Array.from({ length: h }, (_, k) => +(j === k))) }
  const cov = Array.from({ length: h }, () => new Array<number>(h).fill(0))
  for (const r of rows)
    for (let a = 0; a < h; a++) {
      const da = (r[a] ?? 0) - (mean[a] ?? 0)
      const row = cov[a] as number[]
      for (let b = a; b < h; b++) row[b] = (row[b] ?? 0) + (da * ((r[b] ?? 0) - (mean[b] ?? 0))) / Math.max(1, n - 1)
    }
  for (let a = 0; a < h; a++) for (let b = 0; b < a; b++) (cov[a] as number[])[b] = (cov[b] as number[])[a] ?? 0
  const vectors = toFlat(eigh(cov).eigenvectors)
  // Eigenvalues ascend: the last two columns.
  const column = (c: number): number[] => Array.from({ length: h }, (_, j) => vectors[j * h + c] ?? 0)
  return { mean, axes: [column(h - 1), column(h - 2)] }
}

/**
 * A hidden layer's representation of some rows at each snapshot, as embedding steps: each step's activations
 * projected onto their own top two principal axes (raw for one or two units), signs aligned with the previous step so
 * the picture does not flip.
 *
 * @param method - The record's method name.
 * @param snapshots - The snapshots.
 * @param represent - The hidden activations `[n, h]` of the rows under a snapshot's parameters.
 * @param objective - The training loss at each snapshot.
 * @returns The record.
 */
export function representationSteps(
  method: string,
  snapshots: readonly Snapshot[],
  represent: (snapshot: Snapshot) => Tensor,
  objective?: readonly number[],
): EmbeddingSteps {
  let previous: number[][] | undefined
  const positions = snapshots.map((s) => {
    const rows = rowsOf(represent(s))
    const { mean, axes } = principalPlane(rows)
    const projected = rows.map((r) => {
      const centred = r.map((v, j) => v - (mean[j] ?? 0))
      const p = axes.map((a) => a.reduce((acc, w, j) => acc + w * (centred[j] ?? 0), 0))
      return p.length === 1 ? [p[0] ?? 0, 0] : p
    })
    if (previous !== undefined)
      for (let c = 0; c < 2; c++) {
        let dot = 0
        for (const [i, p] of projected.entries()) dot += (p[c] ?? 0) * (previous[i]?.[c] ?? 0)
        if (dot < 0) for (const p of projected) p[c] = -(p[c] ?? 0)
      }
    previous = projected
    return projected
  })
  return embeddingSteps({
    method,
    positions,
    iterations: snapshots.map((s) => s.epoch),
    ...(objective === undefined ? {} : { objective }),
  })
}

/** The training loss (eval mode, all rows) at each snapshot's epoch. */
export function lossAtSnapshots(training: Training): number[] {
  return training.snapshots.map((s) =>
    s.epoch === 0 ? training.initial.loss : (training.evalLoss[s.epoch - 1] ?? Number.NaN),
  )
}

/** How far a training run has got, with the records so far (structurally a hub progress report). */
export interface TrainingProgress {
  /** The fraction of the epochs done. */
  readonly fraction: number
  readonly label: string
  readonly exposition: readonly Exposition[]
}

/**
 * An `onEpoch` callback reporting progress at every snapshot: the records a snapshot draws (the decision surface so
 * far) and the loss curve so far.
 *
 * @param onProgress - Where to report.
 * @param recordsAt - The records of a snapshot.
 * @returns The callback.
 */
export function progressReporter(
  onProgress: (p: TrainingProgress) => void,
  recordsAt: (snapshot: Snapshot) => readonly Exposition[],
): (report: EpochReport) => void {
  const epochs: number[] = []
  const losses: number[] = []
  return (r) => {
    epochs.push(r.epoch)
    losses.push(r.loss)
    if (r.snapshot === undefined) return
    const curve = learningCurve({
      sizes: [...epochs],
      train: [...losses],
      validation: [...losses],
      resource: 'epoch',
      labels: { train: 'training loss', validation: 'training loss' },
    })
    onProgress({
      fraction: r.epoch / Math.max(1, r.epochs),
      label: `epoch ${r.epoch} of ${r.epochs}`,
      exposition: [...recordsAt(r.snapshot), curve],
    })
  }
}
