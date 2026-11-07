/**
 * Fail on type definitions that restate a canonical type instead of using it, so the vector, point, colour and complex
 * types stay singular (`Vec2`, `Vec3`, `Vec4` of `mdml-engine/base`):
 *
 *   node scripts/type-definitions.ts --check
 *
 * 1. A non-exported `type X = Y` where Y is a bare type name is a purposeless local alias: use the name `Y`.
 * 2. A `type` alias whose structure is an unlabelled fixed-size number tuple of two, three or four entries restates
 *    `Vec2`, `Vec3` or `Vec4`: use the canonical type, or a labelled tuple (`readonly [r: number, g: number, ...]`)
 *    when the element names carry meaning.
 * 3. An exported `type A = B` of a bare name needs a doc comment that says what the specialisation adds.
 * 4. A name in `RESERVED` (a generic word that once meant several things: `Box`, `Bounds`, `Schedule`, `Matrix`...) is
 *    defined in its one home only, or nowhere: a second meaning gets a name that says what it is (`BoundVectors`,
 *    `LearningRateSchedule`, `ExprMatrix`). This covers `type`, `interface`, `class` and `enum`, local ones too.
 *
 * An alias of a typed array (`type Vec = Float64Array`) is a local alias like any other: write `Float64Array`.
 * The right-hand side is normalised as `scripts/review/types.mjs` does (comments and `readonly` dropped), through the
 * helpers in `scripts/review/_shared.mjs`. Tests are exempt. A line may opt out with a trailing `// allow: <reason>`.
 */
import { readFileSync } from 'node:fs'
import { join, relative } from 'node:path'
import { aliasLines, isData, isTest, ROOT, walk } from './review/_shared.mjs'

/** Names that are not another definition's name: a bare keyword type is a domain name for a primitive. */
const PRIMITIVE = new Set(
  'string number boolean bigint symbol object unknown never any void null undefined true false'.split(' '),
)
const BARE_NAME = /^[A-Za-z_$][\w$]*(\.[A-Za-z_$][\w$]*)*$/
const TUPLE = /^\[number(,number){1,3}\]$/
const MIN_DOC_WORDS = 4
const DEFINITION =
  /^\s*(?:export\s+)?(?:declare\s+)?(?:(?:type\s+([A-Za-z_$][\w$]*)\s*[<=])|(?:(?:abstract\s+)?(?:interface|class|enum)\s+([A-Za-z_$][\w$]*)))/
const lib = (path: string): string => `packages/engine/src/lib/${path}`

/**
 * Generic names that meant several things in several entry points, with where the one meaning that kept the name
 * lives (a path from the repository root; no home means the name is retired and nothing may define it) and why the
 * others were renamed. A new definition of one of these names is a second meaning: name it for what it is.
 */
const RESERVED: Readonly<Record<string, { readonly homes: readonly string[]; readonly why: string }>> = {
  // Retired: every meaning has a name of its own.
  Matrix: {
    homes: [],
    why: 'rows, a shaped array, a row-major record, symbolic rows: `Rows`, `ShapedArray`, `DenseMatrix`, `ExprMatrix`',
  },
  Mat: { homes: [], why: 'a flat observation matrix, a typed array: `ObservationMatrix`, `Float64Array`' },
  Vec: {
    homes: [],
    why: 'a typed array or a number array: write the type; fixed-size tuples are `Vec2`, `Vec3`, `Vec4`',
  },
  Box: {
    homes: [],
    why: 'bounds, 4-number boxes, placed boxes, bounding boxes: `BoundVectors`, `BoxCoords`, `LayoutBox`, `BoundingBox`, `Aabb`',
  },
  Direction: {
    homes: [],
    why: 'metric, edge and series directions: `MetricDirection`, `EdgeDirection`, `SeriesDirection`',
  },
  Schedule: {
    homes: [],
    why: 'recording, learning-rate, cooling and fidelity schedules: `RecordSchedule`, `LearningRateSchedule`, `CoolingSchedule`, `FidelitySchedule`',
  },
  Selection: {
    homes: [],
    why: 'channels, parents, synthesised features, lambdas: `ChannelSelection`, `ParentSelection`, `SynthesisSelection`, `LambdaSelection`',
  },
  // One home each.
  Bounds: {
    homes: [lib('optim/minimize/objective.ts')],
    why: 'variable bounds; others are `BoundingBox`, `IntervalBounds`, `TruncationBounds`',
  },
  Constraint: {
    homes: [lib('optim/minimize/constrained.ts'), 'packages/render/src/react/interaction/constraints.ts'],
    why: 'scipy constraints (render: a drag constraint); others are `ParameterConstraint`, `HyperparameterConstraint`, `CspConstraint`',
  },
  Kernel: {
    homes: [lib('kernels/kernel.ts')],
    why: 'the kernel object; others are `KernelFunction`, `DiscontinuityKernel`, `RadialKernel`',
  },
  Sampler: {
    homes: [lib('sampling/variance-reduction.ts')],
    why: 'a draw of n samples; others are `StudySampler`, `RowOrder`',
  },
  Transition: {
    homes: [lib('rl/agent.ts')],
    why: 'a step of experience; others are `ChainTransition`, `ConceptTransition`',
  },
  Metric: { homes: [lib('metrics/metric.ts')], why: 'a metric function; others are `PatchMetric`, `DistanceMetric`' },
  Tree: { homes: [lib('tensor/pytree.ts')], why: 'the pytree; others are `ParamTree`, `FittedTree`' },
  Primitive: {
    homes: [lib('tensor/primitive.ts')],
    why: 'the autodiff primitive; the programming one is `GpPrimitive`',
  },
  Shape: { homes: [lib('tensor/shape.ts')], why: 'a tensor shape; the ray-traced one is `RayShape`' },
  Objective: {
    homes: [lib('optim/minimize/objective.ts')],
    why: 'the optimiser objective; the study one is `StudyObjective`',
  },
  Bracket: { homes: [lib('optim/scalar.ts')], why: 'a line bracket; the halving one is `HalvingBracket`' },
  Residual: {
    homes: [lib('nn/module/containers.ts')],
    why: 'the residual connection; others are `ImplicitResidual`, `ResidualNetwork`',
  },
  Option: { homes: [lib('space/tables.ts')], why: 'an option of a table; the agents one is `HierarchicalOption`' },
  Knobs: { homes: [lib('space/tables.ts')], why: 'the knob reader; the generator record is `GeneratorKnobs`' },
  Dataset: { homes: [lib('data/schema/common.ts')], why: 'the dataset union; the agents one is `TransitionDataset`' },
  TrajectoryData: {
    homes: [lib('data/schema/trajectories.ts')],
    why: 'the data schema; the decision transformer one is `DecisionTransformerData`',
  },
  Importances: {
    homes: [lib('exposition/attributions.ts')],
    why: 'the exposition record; the tune one is `ParamImportances`',
  },
  ConfusionMatrix: {
    homes: [lib('exposition/evaluation.ts')],
    why: 'the exposition record; the metric result is `LabelledConfusionMatrix`',
  },
  LearningCurve: {
    homes: [lib('exposition/searches.ts')],
    why: 'the exposition record; the rl input is `LearningCurveInput`',
  },
  Projected: { homes: [lib('scene/projection.ts')], why: 'a projected point; the splats are `ProjectedGaussians`' },
  QuadraticModel: {
    homes: [lib('optim/minimize/trust-subproblems.ts')],
    why: 'the trust-region model; the field one is `LocalModels`',
  },
  GlmProblem: { homes: [lib('optim/glm.ts')], why: 'the smooth GLM problem; the design one is `GlmDesignProblem`' },
  VectorField: {
    homes: [lib('exposition/fields.ts')],
    why: 'the exposition record; the callables are `OdeVectorField`, `PointVectorField`',
  },
  StabilityRegion: {
    homes: [lib('exposition/stability-region.ts')],
    why: 'the exposition record; the sampled one is `SampledStabilityRegion`',
  },
  AffineMap: {
    homes: [lib('numerics/geometry/transforms.ts')],
    why: 'a d-dimensional map; the planar one is `PlanarAffineMap`',
  },
  FieldDerivatives: {
    homes: [lib('losses/physics.ts')],
    why: 'a field and its derivatives; the pde one is `CollocationDerivatives`',
  },
  HookContext: {
    homes: [lib('trace/contract.ts')],
    why: 'the trace hook context; the nn one is `TrainingHookContext`',
  },
  Proposal: {
    homes: [lib('numerics/quadrature/monte-carlo.ts')],
    why: 'an importance proposal; the chain one is `ChainProposal`',
  },
  Transform: { homes: [lib('scene/types.ts')], why: 'a scene transform; the feature one is `FeatureArithmetic`' },
}

const problems: string[] = []
for (const root of ['packages', 'apps']) {
  for (const file of walk(join(ROOT, root))) {
    if (isTest(file) || isData(file)) continue
    const where = relative(ROOT, file)
    const text = readFileSync(file, 'utf8')
    text.split('\n').forEach((line, i) => {
      const found = DEFINITION.exec(line)
      const name = found?.[1] ?? found?.[2]
      const reserved = name === undefined ? undefined : RESERVED[name]
      if (name === undefined || reserved === undefined || reserved.homes.includes(where) || line.includes('// allow:'))
        return
      const home = reserved.homes.length === 0 ? 'retired' : `defined in ${reserved.homes.join(' and ')}`
      problems.push(`${where}:${i + 1}: \`${name}\` is ${home} (${reserved.why}); name this one for what it is`)
    })
    for (const a of aliasLines(text)) {
      if (a.optOut) continue
      const bare = !a.generic && BARE_NAME.test(a.shape) && !PRIMITIVE.has(a.shape)
      if (TUPLE.test(a.shape)) {
        const size = a.shape.split(',').length
        problems.push(
          `${where}:${a.line}: \`${a.name}\` restates Vec${size}; use it, or a labelled tuple if the names matter`,
        )
      } else if (bare && !a.exported) {
        problems.push(`${where}:${a.line}: \`${a.name}\` is a local alias of \`${a.shape}\`; use \`${a.shape}\``)
      } else if (bare && a.doc.split(' ').filter((w) => /\w/.test(w)).length < MIN_DOC_WORDS) {
        problems.push(`${where}:${a.line}: \`${a.name}\` aliases \`${a.shape}\`; a doc comment must say what it adds`)
      }
    }
  }
}
if (problems.length > 0) {
  for (const p of problems) console.error(p)
  process.exit(1)
}
console.log('type-definitions: clean')
