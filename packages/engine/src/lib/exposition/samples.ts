/**
 * One small valid record of every exposition kind, built by the kind's own constructor: the fixtures of the contract
 * test over all kinds, and of the smoke tests of the views that draw them. Each is the smallest legal input of its
 * constructor, with finite numbers only.
 */
import { circle, scene, structure } from '../scene/index.ts'
import { agentSteps } from './agents.ts'
import { anomalyScores } from './anomaly.ts'
import { featureAttributions, importances } from './attributions.ts'
import { audioClip } from './audio.ts'
import { boardSteps } from './boards.ts'
import { valueGrids } from './cells.ts'
import { chainDiagnostics } from './chain-diagnostics.ts'
import { clusteringSteps, mixtureSteps } from './clustering.ts'
import { complexPlane } from './complex-plane.ts'
import { conceptScores } from './concepts.ts'
import { curveSet } from './curves.ts'
import { dataBias } from './data-bias.ts'
import { dependence } from './dependence.ts'
import { diagramRecord } from './diagrams.ts'
import { componentsRecord, embeddingSteps } from './embeddings.ts'
import { ensembleStages } from './ensembles.ts'
import { epicycles } from './epicycles.ts'
import { episodeRecord } from './episodes.ts'
import { equationFit } from './equation-fit.ts'
import { confusionMatrix } from './evaluation.ts'
import { counterfactualSet, prototypeSet } from './examples.ts'
import { explanationQuality } from './explanation-quality.ts'
import { groupReport, groupReportFromTables, tableRates } from './fairness.ts'
import { vectorFieldRecord } from './fields.ts'
import { paretoFront } from './frontiers.ts'
import { imageSteps } from './images.ts'
import { landscapeSlice } from './landscape-slice.ts'
import { coefficientTable, crossValidationCurve, regularisationPath } from './linear.ts'
import { marginRecord } from './margins.ts'
import { modelRun } from './model-run.ts'
import { likelihoodTable, neighbourRecord } from './neighbours.ts'
import { pipelineStages } from './pipelines.ts'
import { pointsSteps } from './points-steps.ts'
import { populationSteps } from './populations.ts'
import { hyperparameterSurface, posteriorGrid } from './posteriors.ts'
import { reachabilityRecord } from './reachability.ts'
import type { Exposition, ExpositionKind } from './record.ts'
import { ruleSet } from './rules.ts'
import { sampleGrid } from './sample-grid.ts'
import { sbcRun } from './sbc-run.ts'
import { sceneSteps } from './scene-steps.ts'
import { learningCurve, searchTrials } from './searches.ts'
import { stateSequence } from './sequences.ts'
import { simplexRecord } from './simplex.ts'
import { stabilityRegion } from './stability-region.ts'
import { stripRecord } from './strips.ts'
import { structureSteps } from './structure-steps.ts'
import { subgroupReport } from './subgroups.ts'
import { decisionSurface } from './surfaces.ts'
import { markedTable } from './tables.ts'
import { thresholdSweep } from './thresholds.ts'
import { transcript } from './transcript.ts'
import { pruningPath, treeRecord } from './trees.ts'
import { calibrationRecord, conformalRecord } from './trust.ts'

const network = structure(
  'net',
  [
    { id: 'x', label: 'x', role: 'input' },
    { id: 'y', label: 'y', role: 'output' },
  ],
  [{ from: 'x', to: 'y' }],
)
const table = [
  [1, 0],
  [0, 1],
]
const zeroRates = {
  selectionRate: 0,
  truePositiveRate: 0,
  falsePositiveRate: 0,
  precision: 0,
  accuracy: 0,
  baseRate: 0,
}
const worstGap = { subgroup: 0, gap: 0 }
const worstGaps = {
  selectionRate: worstGap,
  truePositiveRate: worstGap,
  falsePositiveRate: worstGap,
  precision: worstGap,
  accuracy: worstGap,
  baseRate: worstGap,
}

/**
 * A small valid record of each exposition kind, made by a function so that every call returns fresh arrays.
 *
 * @example
 * ```ts
 * import { expositionSamples, isExposition } from 'omai-engine/exposition'
 * isExposition(expositionSamples.curves()) // => true
 * expositionSamples.thresholdSweep().kind // => 'thresholdSweep'
 * ```
 */
export const expositionSamples: { readonly [K in ExpositionKind]: () => Extract<Exposition, { kind: K }> } = {
  regularisationPath: () =>
    regularisationPath({
      penalties: [1, 0.1],
      features: ['a', 'b'],
      coefficients: [
        [0, 0],
        [0.4, 0],
      ],
    }),
  coefficientTable: () => coefficientTable({ names: ['x'], estimates: [2], standardErrors: [0.1] }),
  crossValidationCurve: () =>
    crossValidationCurve({
      parameter: 'alpha',
      values: [1, 2],
      meanScores: [0.8, 0.7],
      standardErrors: [0.1, 0.1],
      higherIsBetter: true,
    }),
  tree: () =>
    treeRecord({
      criterion: 'gini',
      features: ['x'],
      nodes: [{ id: 0, parent: null, depth: 0, rows: [0, 1], value: [1, 1], impurity: 0.5, candidates: [], order: 0 }],
    }),
  pruningPath: () => pruningPath({ alphas: [0, 0.1], impurities: [0, 0.2], leaves: [4, 1] }),
  ensembleStages: () => ensembleStages({ method: 'adaboost', weights: [0.8, 0.5], trainScores: [0.7, 0.85] }),
  margin: () => marginRecord({ support: [0, 3], dualCoefficients: [0.5, -0.5], intercept: 0.1, margins: [1, 2, 3, 1] }),
  neighbours: () => neighbourRecord({ indices: [[2, 0]], distances: [[0.1, 0.4]], metric: 'euclidean' }),
  likelihoodTable: () =>
    likelihoodTable({ classes: ['a', 'b'], features: ['x'], logPriors: [-0.7, -0.7], contributions: [[-1], [-2]] }),
  posteriorGrid: () => posteriorGrid({ inputs: [[0], [1]], mean: [0, 1], variance: [1, 0.5] }),
  hyperparameterSurface: () =>
    hyperparameterSurface({ parameters: ['length', 'noise'], x: [1, 2], y: [0.1], values: [[-3, -2]] }),
  clusteringSteps: () =>
    clusteringSteps({
      method: 'kmeans',
      assignments: [
        [0, 0, 1],
        [0, 1, 1],
      ],
      objective: [3, 1],
    }),
  mixtureSteps: () =>
    mixtureSteps({ family: 'gaussian', weights: [[0.5, 0.5]], means: [[[0], [1]]], objective: [-10] }),
  components: () =>
    componentsRecord({ method: 'pca', features: ['x', 'y'], components: [[0.6, 0.8]], explainedVarianceRatio: [0.9] }),
  embeddingSteps: () =>
    embeddingSteps({
      method: 'tsne',
      positions: [
        [
          [0, 0],
          [1, 1],
        ],
      ],
      iterations: [0],
    }),
  anomalyScores: () => anomalyScores({ method: 'isolationForest', scores: [0.4, 0.9], threshold: 0.6 }),
  decisionSurface: () => decisionSurface({ features: ['x0', 'x1'], x: [0, 1], y: [0], labels: [[0, 1]] }),
  pipelineStages: () => pipelineStages({ stages: [] }),
  searchTrials: () =>
    searchTrials({
      method: 'grid',
      parameters: ['alpha'],
      trials: [{ params: { alpha: 1 }, value: 0.8, state: 'complete' }],
      higherIsBetter: true,
      best: 0,
    }),
  learningCurve: () => learningCurve({ sizes: [10, 20], train: [1, 0.95], validation: [0.6, 0.8] }),
  episodes: () => episodeRecord({ environment: 'CartPole', agent: 'dqn', returns: [12, 40], lengths: [12, 40] }),
  boardSteps: () =>
    boardSteps({
      method: 'bfs',
      layout: { type: 'grid', rows: 1, columns: 3 },
      categories: ['empty', 'explored', 'frontier'],
      steps: [{ cells: [2, 0, 0] }, { cells: [1, 2, 0] }],
      metrics: [{ name: 'expanded', values: [0, 1] }],
    }),
  populationSteps: () =>
    populationSteps({
      method: 'ga',
      generations: [
        { fitness: [1, 3], best: { fitness: 3, label: '11' } },
        { fitness: [3, 4], best: { fitness: 4 } },
      ],
    }),
  curves: () =>
    curveSet({
      x: [1, 2, 3],
      xLabel: 'epoch',
      series: [
        { name: 'sgd', y: [1, 0.6, 0.4] },
        { name: 'adam', y: [0.8, 0.3, 0.2] },
      ],
    }),
  diagram: () => diagramRecord({ structure: network, steps: [{ overlay: { active: ['x'] }, caption: 'forward' }] }),
  valueGrids: () =>
    valueGrids({
      grids: [
        {
          name: 'filter 1',
          values: [
            [1, 0],
            [0, -1],
          ],
        },
      ],
      diverging: true,
    }),
  confusionMatrix: () =>
    confusionMatrix({
      classes: ['cat', 'dog'],
      counts: [
        [5, 1],
        [2, 7],
      ],
    }),
  stateSequence: () =>
    stateSequence({
      states: ['fair', 'loaded'],
      posteriors: [
        [0.9, 0.1],
        [0.2, 0.8],
      ],
      decoded: [0, 1],
    }),
  imageSteps: () =>
    imageSteps({
      method: 'vae',
      height: 1,
      width: 2,
      groups: ['data', 'samples'],
      steps: [{ iteration: 0, images: [[[0, 1]], [[0.5, 0.5]]] }],
    }),
  epicycles: () =>
    epicycles({
      method: 'fourier series',
      source: 'circle',
      target: { x: [1, 0, -1, 0], y: [0, 1, 0, -1] },
      centroid: [0, 0],
      chain: { re: [1], im: [0], k: [1] },
      trace: { x: [1, 0, -1, 0, 1], y: [0, 1, 0, -1, 0] },
      spectrum: { k: [1], magnitude: [1], kept: [true] },
      energyKept: 1,
      steps: 4,
    }),
  agentSteps: () =>
    agentSteps({
      method: 'boids',
      world: { width: 10, height: 10, wrap: true },
      frames: [{ t: 0, x: [1, 2], y: [3, 4], heading: [0, 1.5] }],
      series: [{ name: 'polarisation', values: [0.2] }],
    }),
  featureAttributions: () =>
    featureAttributions({
      method: 'exact',
      features: ['a', 'b'],
      output: 'prediction',
      baseValue: 1,
      values: [[2, -0.5]],
      data: [[3, 4]],
      predictions: [2.5],
    }),
  importances: () =>
    importances({
      method: 'permutation',
      features: ['a', 'b'],
      mean: [0.3, 0.01],
      spread: [0.05, 0.01],
      metric: 'accuracy',
      direction: 'higher',
    }),
  dependence: () =>
    dependence({
      method: 'partial dependence',
      features: ['age', 'bmi'],
      output: 'prediction',
      curves: [
        {
          feature: 1,
          grid: [0, 1],
          average: [2, 3],
          individual: [
            [1, 2],
            [3, 4],
          ],
        },
      ],
    }),
  counterfactuals: () =>
    counterfactualSet({
      method: 'growing spheres',
      features: ['income', 'debt'],
      output: 'P(approve)',
      original: [30, 10],
      originalPrediction: 0.3,
      counterfactuals: [{ x: [36, 10], prediction: 0.6, distance: 6, valid: true, changed: [0] }],
    }),
  prototypes: () =>
    prototypeSet({
      method: 'MMD-critic',
      features: ['a'],
      prototypes: [0],
      criticisms: [2],
      prototypeData: [[0]],
      criticismData: [[9]],
      witness: [-0.1, -0.1, 0.3],
    }),
  concepts: () =>
    conceptScores({
      method: 'TCAV',
      target: 'zebra',
      layer: 'mixed4c',
      concepts: [{ name: 'striped', scores: [0.9, 0.8], randomScores: [0.5, 0.4], mean: 0.85, pValue: 0.01 }],
    }),
  rules: () =>
    ruleSet({
      method: 'anchors',
      features: ['age'],
      output: 'P(y)',
      rules: [{ clauses: [{ feature: 0, op: '>', threshold: 40 }], precision: 0.97, coverage: 0.3 }],
    }),
  explanationQuality: () =>
    explanationQuality({
      explainers: ['SHAP', 'LIME'],
      metrics: [{ name: 'deletion AUC', category: 'faithfulness', values: [0.2, 0.3], better: 'lower' }],
    }),
  groupReport: () =>
    groupReport(
      groupReportFromTables({
        attribute: 'sex',
        groups: ['m', 'f'],
        tables: [
          [
            [5, 1],
            [1, 3],
          ],
          [
            [4, 0],
            [3, 2],
          ],
        ],
      }),
    ),
  calibration: () =>
    calibrationRecord({
      nBins: 2,
      strategy: 'uniform',
      curves: [{ name: 'model', meanPredicted: [0.2, 0.8], observed: [0.1, 0.9], counts: [5, 5], ece: 0.1, mce: 0.1 }],
    }),
  conformal: () =>
    conformalRecord({
      task: 'regression',
      method: 'split',
      alpha: 0.1,
      coverage: 0.9,
      x: [0, 1],
      y: [1, 2],
      lower: [0, 1],
      upper: [2, 3],
    }),
  thresholdSweep: () =>
    thresholdSweep({
      classes: ['no', 'yes'],
      thresholds: [0.2, 0.8, 1],
      positives: 1,
      negatives: 1,
      tp: [1, 1, 0],
      fp: [1, 0, 0],
      tn: [0, 1, 1],
      fn: [0, 0, 1],
      tpr: [1, 1, 0],
      fpr: [1, 0, 0],
      precision: [0.5, 1, 1],
      selectionRate: [1, 0.5, 0],
      accuracy: [0.5, 1, 0.5],
      rocAuc: 1,
      averagePrecision: 1,
    }),
  subgroupReport: () =>
    subgroupReport({
      attributes: ['sex'],
      values: [['f']],
      subgroups: [['f']],
      counts: [2],
      tables: [table],
      rates: [tableRates(table)],
      overall: tableRates(table),
      minSize: 1,
      gaps: zeroRates,
      worst: worstGaps,
    }),
  dataBias: () =>
    dataBias({
      attribute: 'sex',
      groups: ['f', 'm'],
      reference: 1,
      counts: [40, 60],
      shares: [0.4, 0.6],
      baseRates: [0.3, 0.5],
      overallBaseRate: 0.42,
      disparities: [],
    }),
  paretoFront: () =>
    paretoFront({
      violationName: 'gap',
      families: [{ name: 'grid', violation: [0.1, 0.2], accuracy: [0.7, 0.8], front: [0, 1] }],
    }),
  vectorField: () =>
    vectorFieldRecord({
      x: [-1, 1],
      y: [-1, 1],
      u: [
        [1, 1],
        [1, 1],
      ],
      v: [
        [0, 0],
        [0, 0],
      ],
      fixedPoints: [{ at: [0, 0], stability: 'saddle' }],
    }),
  strip: () =>
    stripRecord({
      tokens: ['the', 'cat', 'sat'],
      tracks: [
        { name: 'noun', values: [0.1, 0.9, 0.2] },
        { name: 'verb', values: [0.1, 0.05, 0.8] },
      ],
      path: [0, 0, 1],
      marks: [2],
    }),
  markedTable: () =>
    markedTable({
      columns: ['test', 'p'],
      cells: [
        ['t', 0.03],
        ['z', 0.4],
      ],
      marks: [{ row: 0, status: 'decision', note: 'p < 0.05' }],
    }),
  structureSteps: () => structureSteps({ structure: network, steps: [{ step: 0, nodes: { x: { value: 1 } } }] }),
  pointsSteps: () =>
    pointsSteps({
      dims: 2,
      steps: [
        { step: 0, points: [[0, 0]] },
        {
          step: 1,
          points: [
            [1, 1],
            [2, 0],
          ],
        },
      ],
    }),
  complexPlane: () =>
    complexPlane({
      domain: 's',
      poles: [
        [-1, 2],
        [-1, -2],
      ],
      zeros: [[-3, 0]],
    }),
  simplex: () =>
    simplexRecord({
      shape: 'triangle',
      labels: ['rock', 'paper', 'scissors'],
      paths: [
        {
          name: 'run',
          points: [
            [0.5, 0.3, 0.2],
            [0.4, 0.4, 0.2],
          ],
        },
      ],
    }),
  reachability: () => reachabilityRecord({ reach: [null, 0.1, 0.1, 0.9, 0.1], threshold: 0.5 }),
  sceneSteps: () => sceneSteps({ steps: [{ step: 0, scene: scene(40, 40, [circle(20, 20, 5)]) }] }),
  landscapeSlice: () =>
    landscapeSlice({ dims: 1, axes: ['step'], x: [0, 1, 2], curve: [3, 1, 2], paths: [{ coords: [[0], [1]] }] }),
  stabilityRegion: () =>
    stabilityRegion({
      method: 'm',
      x: [-1, 0],
      y: [0, 1],
      modulus: [
        [0.5, 1],
        [2, 1],
      ],
      marks: [{ at: [-0.5, 0] }],
    }),
  equationFit: () =>
    equationFit({ terms: [{ tex: 'x', estimate: 2, truth: 2 }], x: [0, 1], fit: [0, 2], truth: [0, 2] }),
  // A second of a 100 Hz tone: long enough for the view's spectrogram to have a window.
  audio: () =>
    audioClip({
      sampleRate: 1024,
      samples: Array.from({ length: 1024 }, (_, i) => Math.sin((2 * Math.PI * 100 * i) / 1024)),
    }),
  transcript: () =>
    transcript({
      tokens: [{ text: 'Hi' }, { text: ' there' }],
      tracks: [{ name: 'head', weights: [0.3, 0.7] }],
      emitted: 1,
    }),
  sampleGrid: () => sampleGrid({ modality: 'text', steps: [{ step: 0, samples: ['xqz', 'ab'] }] }),
  modelRun: () =>
    modelRun({
      structure: network,
      status: 'running',
      series: [
        { name: 'change', label: 'largest change of a belief', steps: [1, 2], values: [0.5, 0.1], direction: 'lower' },
      ],
      steps: [
        {
          step: 1,
          nodes: { x: { mean: 0.5, sd: 1, text: '0.5 ± 1' } },
          messages: [{ from: 'x', to: 'y', text: 'N(μ = 0.5, σ² = 1)' }],
        },
        { step: 2, nodes: { y: { observed: 1, text: 'observed 1' } } },
      ],
    }),
  chainDiagnostics: () =>
    chainDiagnostics(
      {
        mu: [
          [0.1, 0.4, 0.2, 0.5],
          [0.3, 0.2, 0.6, 0.4],
        ],
      },
      {
        bins: 2,
        divergent: [
          [0, 0, 1, 0],
          [0, 0, 0, 0],
        ],
      },
    ),
  sbcRun: () =>
    sbcRun({
      replications: 10,
      numRanks: 3,
      alpha: 0.01,
      passes: true,
      quantities: [
        {
          name: 'mu',
          ranks: [0, 1, 2, 3, 0, 1, 2, 3, 1, 2],
          numRanks: 3,
          histogram: [2, 3, 3, 2],
          chiSquare: { statistic: 0.4, pValue: 0.94 },
          ecdf: { statistic: 0.05, pValue: 0.9 },
          pValue: 1,
          diagnosis: 'uniform',
          passes: true,
        },
      ],
    }),
}
