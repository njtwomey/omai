import { classificationLosses, penalties } from 'mdml-engine/losses'
import { linearClassifier } from 'mdml-engine/methods/linear'
import { smoothSolverConfig, smoothSolvers } from 'mdml-engine/optim'
import { type DictSpace, defaults, dict, type SpaceValue } from 'mdml-engine/space'
import type { Tensor } from 'mdml-engine/tensor'
import type { Config } from 'mdml-engine/trace'
import {
  Choice,
  ControlLayout,
  ControlPanel,
  DatasetPicker,
  type DatasetSelection,
  modeSlot,
  type Panel,
  SessionPanels,
  SessionPlayer,
  type SlotKnobs,
  SlotVariants,
  SpaceControls,
  selectEntry,
  tableRules,
  tableSlot,
  useDataset,
  useSessionPreferences,
  useSessions,
  variantsOfSlot,
} from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Linear classifier, every solver',
  question: 'What does a method page look like: one objective, every solver, one player?',
  explain:
    'The template for the pages of the encyclopedia. A `Method` declares its slots (here a loss, a penalty and a solver); the page fills the objective from the option tables, runs one session per solver with `useSessions` and `variantsOfSlot`, and draws the same tape three ways under one cursor: curves overlaid, the decision surface of each solver side by side, and per-row margins. Controls come from the slots: a dataset picker, `SpaceControls` over the loss and penalty (options that would stall a gradient method are listed greyed, with the reason), and `SlotVariants` for the solvers, each with its own knobs.',
}

// The page's configuration: the objective is the model (a loss and a penalty); the solver is the mode, one variant each.
const OBJECTIVE: DictSpace = dict({ loss: classificationLosses.space, penalty: penalties.space })
const STEPS = 150
// Margins are drawn on a fixed scale, so rows keep their colour from one step and one solver to the next.
const MARGIN_RANGE = [-6, 6] as const

// The objective a page starts from: logistic loss with a ridge penalty (without a penalty, separable data sends the
// weights to infinity and every solver "converges" only by running out of gradient).
const objectiveDefaults = (): Config => ({
  loss: { kind: 'logistic' },
  penalty: { kind: 'l2', ...(defaults(penalties.space.options['l2'] as DictSpace) as object) },
})

// The stochastic-gradient mode is Adam at a learning rate that suits this problem; every other solver starts at its
// own defaults (a page's reset puts the knobs back here).
const solverDefaults = (): SlotKnobs => {
  const { kind: _kind, ...adam } = smoothSolverConfig({
    kind: 'gradient',
    optimiser: { kind: 'adam' },
    learningRate: 0.1,
  }) as Record<string, SpaceValue>
  return { gradient: adam }
}

// What rules a loss or a penalty out: the solvers of this slot follow the gradient, so the objective must be smooth
// (the tables' own `smooth` property). The rule only greys; the controls offer the whole table.
const OBJECTIVE_RULE = tableRules({
  loss: tableSlot(classificationLosses, (p) =>
    p.smooth ? undefined : 'not smooth: a gradient has no direction at its kink, and the solvers here would stall',
  ),
  penalty: tableSlot(penalties, (p) =>
    p.smooth ? undefined : 'not smooth: the kink at zero stalls a gradient method (a proximal solver handles it)',
  ),
})

const describe = (name: string): string => smoothSolvers.modeOf({ kind: name }).description ?? ''

const optimiserOf = (knobs: SlotKnobs): string => {
  const o = knobs['gradient']?.['optimiser']
  return typeof o === 'object' && o !== null && !Array.isArray(o) ? String((o as { kind?: unknown }).kind) : 'sgd'
}

type Rows = 'surface' | 'margins'

// The layout: what the tape is drawn as. Curves of the tape overlay the variants (one colour each); every other panel
// is drawn once per variant, side by side.
function layoutFor(rows: Rows): readonly Panel[] {
  return [
    { kind: 'curves', paths: ['solver/objective'], title: 'objective (click to scrub)' },
    {
      kind: 'curves',
      paths: ['solver/gradNorm'],
      log: 'y',
      title: 'gradient norm, log scale: first-order methods decay, curvature methods fall away',
    },
    { kind: 'curves', paths: ['model/accuracy'], title: 'accuracy on the rows' },
    { kind: 'curves', paths: ['solver/step'], log: 'y', title: 'length of each step, log scale' },
    rows === 'surface'
      ? { kind: 'rows', overlay: 'surface', title: 'decision surface' }
      : {
          kind: 'rows',
          title: 'margin of each row (colour), the misclassified ones ringed',
          values: { path: 'model/margins', diverging: true, range: MARGIN_RANGE, label: 'margin' },
          marks: { path: 'model/errors', names: { misclassified: 'misclassified' } },
        },
  ]
}

export default function LinearClassifierPage(): ReactNode {
  const [selection, setSelection] = useState<DatasetSelection>(() => selectEntry('classification2d', 'moons'))
  const { data } = useDataset('classification2d', selection)
  if (!('x' in data) || !('y' in data)) return <p>This page needs a two-class data set.</p>
  return <Page x={data.x} y={data.y} selection={selection} setSelection={setSelection} />
}

function Page(props: {
  readonly x: Tensor
  readonly y: Tensor
  readonly selection: DatasetSelection
  readonly setSelection: (s: DatasetSelection) => void
}): ReactNode {
  const { x, y, selection, setSelection } = props
  const [objective, setObjective] = useState<Config>(objectiveDefaults)
  const [knobs, setKnobs] = useState<SlotKnobs>(solverDefaults)
  const [rows, setRows] = useState<Rows>('surface')
  const [at, setAt] = useState<number | null>(null)
  const prefs = useSessionPreferences()
  // The method is made over the data; its slots are the page's controls.
  const method = useMemo(() => linearClassifier({ x, y }), [x, y])
  // The solver slot: whether each option applies (the mode table's `admissible`, given the other slots).
  const solverRule = modeSlot(smoothSolvers, {
    loss: (objective['loss'] as { kind: string }).kind,
    penalty: (objective['penalty'] as { kind: string }).kind,
  })
  // One variant per solver that applies, each with the knobs the page holds for it; the objective is shared.
  const variants = variantsOfSlot(
    method.config,
    'solver',
    {},
    {
      knobs,
      admissible: (n) => solverRule.reason(n) === undefined,
      label: (n) => (n === 'gradient' ? `gradient (${optimiserOf(knobs)})` : n),
    },
  )
  const live = useSessions(
    method,
    { config: objective, mode: prefs.mode, steps: STEPS },
    variants,
    [method, JSON.stringify(objective)],
    { hold: prefs.hold },
  )
  const reset = (): void => {
    setObjective(objectiveDefaults())
    setKnobs(solverDefaults())
    setRows('surface')
    setSelection(selectEntry('classification2d', 'moons'))
  }
  const controls = (
    <div className="grid grid-cols-1 items-start gap-3 md:grid-cols-2 xl:grid-cols-[1fr_1fr_1.15fr_0.7fr]">
      <ControlPanel title="data" columns={1}>
        <DatasetPicker collection="classification2d" value={selection} onChange={setSelection} exclude={['n']} />
      </ControlPanel>
      <ControlPanel title="objective: the model" columns={1}>
        <SpaceControls
          space={OBJECTIVE}
          value={objective}
          onChange={(v) => setObjective(v as Config)}
          admissible={OBJECTIVE_RULE.admissible}
          {...(OBJECTIVE_RULE.why === undefined ? {} : { why: OBJECTIVE_RULE.why })}
        />
      </ControlPanel>
      <ControlPanel title="solvers: the modes" columns={1}>
        <SlotVariants
          space={smoothSolvers.space}
          knobs={knobs}
          onChange={setKnobs}
          label="one variant each"
          descriptions={Object.fromEntries(smoothSolvers.names.map((n) => [n, describe(n)]))}
          admissible={(n) => solverRule.reason(n) === undefined}
          why={solverRule.reason}
        />
      </ControlPanel>
      <ControlPanel title="view" columns={1}>
        <Choice
          label="rows panel"
          value={rows}
          options={[
            { value: 'surface', label: 'surface' },
            { value: 'margins', label: 'margins' },
          ]}
          onChange={setRows}
        />
      </ControlPanel>
    </div>
  )
  return (
    <ControlLayout controls={controls} placement="top">
      <div className="flex flex-col gap-4">
        <Caption />
        <SessionPlayer live={live} at={at} onScrub={setAt} onReset={reset} />
        <SessionPanels
          id="gallery/linear-classifier"
          live={live}
          at={at}
          onScrub={setAt}
          layout={layoutFor(rows)}
          x={x}
          y={y}
        />
      </div>
    </ControlLayout>
  )
}

// The page explains itself: what is held fixed, what differs, and how to read the panels.
function Caption(): ReactNode {
  return (
    <aside className="flex flex-col gap-2 rounded-lg border border-grid bg-surface p-4 text-sm leading-relaxed text-fg-secondary">
      <p>
        <strong className="text-fg">One objective, minimised seven ways.</strong> The loss and penalty above are the
        model; every solver listed above starts from the same weights on the same rows, so they all end at the same line
        and the surfaces below converge to one picture. What differs is the road.
      </p>
      <p>
        Gradient descent steps downhill with a line search; conjugate gradient and (L-)BFGS reuse past gradients to bend
        the direction toward the curvature; Newton solves for the exact curvature each step and lands in a handful;
        coordinate descent fixes all weights but one. The stochastic mode takes cheap fixed-size steps at a learning
        rate, so it keeps moving after the others have stopped. Read the gradient-norm panel: a solver is finished when
        its curve reaches the tolerance, and the steepness of that fall is its rate. Press play, or drag the scrubber,
        and click any curve to jump there.
      </p>
    </aside>
  )
}
