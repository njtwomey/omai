/**
 * The plain data the probit lab passes between its workers and its page. Everything here is structured-cloneable (numbers,
 * strings, arrays and typed arrays), so a worker's result crosses as it is and the page only draws it.
 */

/** The datasets the lab offers: the library's two-dimensional classification generators. */
export type DatasetId = 'moons' | 'circles' | 'xor' | 'spirals' | 'blobs'

/** What the user chose about the problem. */
export interface ProblemSettings {
  readonly dataset: DatasetId
  /** Rows in all, before the split. */
  readonly rows: number
  /** The noise of the generator (its own scale). */
  readonly noise: number
  readonly seed: number
  /** The degree of the polynomial feature map. */
  readonly degree: number
  /** The standard deviation of the prior of each (unit-scale) coefficient. */
  readonly priorScale: number
}

/** Labelled points of the plane. */
export interface Points {
  readonly x: readonly (readonly number[])[]
  readonly y: readonly number[]
}

/** The problem a run solves: the points split into training and held-out rows, and the rectangle the boundary is drawn over. */
export interface Problem {
  readonly settings: ProblemSettings
  readonly train: Points
  readonly test: Points
  readonly extent: { readonly x: readonly [number, number]; readonly y: readonly [number, number] }
}

/** How a run's draws are organised: a sampler's chains grow, an approximation is redrawn at every snapshot. */
export type Organisation = 'chain' | 'iterate'

/** What the worker is asked: one method on one problem. */
export interface LabRequest {
  readonly problem: Problem
  /** The id of a method of `METHODS`. */
  readonly method: string
  readonly budgetMs: number
  readonly seed: number
  /** The side of the predictive grid of each frame. */
  readonly gridSize: number
}

/** The way a run ended. */
export type Ending = 'converged' | 'completed' | 'budget' | 'stopped' | 'diverged' | 'failed'

/** What a held-out set says of a method. */
export interface Scores {
  readonly accuracy: number
  readonly logPredictive: number
}

/** Summary of one coefficient under one method. */
export interface CoefficientSummary {
  readonly mean: number
  readonly sd: number
  readonly lower: number
  readonly upper: number
  readonly rhat: number | undefined
  readonly ess: number | undefined
}

/**
 * The draws a method produced, as one pool: draw `i` is a row of `values` (`p` columns), shown from `at[i]` to `until[i]`
 * (milliseconds of the run). A sampler's draws appear as they are made and stay (`until` infinite), an approximation's draws
 * are those of one snapshot and are shown until the next. `chain` and `order` place a sampler's draw in its chain.
 */
export interface DrawPool {
  readonly p: number
  readonly values: Float64Array
  readonly at: Float64Array
  readonly until: Float64Array
  readonly chain: Int32Array
  readonly order: Int32Array
}

/** One snapshot of a run, at the time it was taken. */
export interface LabFrame {
  readonly step: number
  readonly elapsedMs: number
  /** The mean and the standard deviation of each coefficient at the snapshot. */
  readonly mean: Float64Array
  readonly sd: Float64Array
}

/** A curve over the steps of the run (the ELBO of a variational fit, the log density of an optimiser). */
export interface RunCurve {
  readonly label: string
  readonly direction: 'higher' | 'lower'
  readonly steps: Float64Array
  readonly values: Float64Array
}

/** What a worker returns for one method. */
export interface MethodResult {
  readonly id: string
  readonly label: string
  readonly organisation: Organisation
  /** Whether the approximation is a normal (its marginals are drawn as normals). */
  readonly gaussian: boolean
  readonly ending: Ending
  readonly message: string | undefined
  readonly steps: number
  readonly elapsedMs: number
  readonly names: readonly string[]
  readonly frames: readonly LabFrame[]
  /** The predictive probability on the grid at each frame, `frames.length` grids of `gridSize ** 2`, rows of the grid by ordinate. */
  readonly grids: Float32Array
  readonly gridSize: number
  readonly pool: DrawPool
  readonly curve: RunCurve | undefined
  readonly coefficients: readonly CoefficientSummary[]
  readonly train: Scores
  readonly test: Scores
  /** The worst R-hat and the smallest effective sample size over the coefficients, for a sampler. */
  readonly maxRhat: number | undefined
  readonly minEss: number | undefined
  readonly divergences: number | undefined
  /** The draws made after warm-up, for a sampler. */
  readonly draws: number | undefined
}

/** What a worker says while it works. */
export interface LabStatus {
  readonly type: 'status'
  readonly phase: 'running' | 'finishing'
  readonly step: number
  readonly elapsedMs: number
}
