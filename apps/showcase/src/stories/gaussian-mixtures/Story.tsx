import { type ReactNode, useState } from 'react'
import { Link } from '../../../../site/src/shell/Link.tsx'
import { Aside, Code, Figure } from '../../kit/Blocks.tsx'
import { Choice, type ChoiceOption } from '../../kit/Choice.tsx'
import { Equation, Tex } from '../../kit/Maths.tsx'
import { Play } from '../../kit/Play.tsx'
import { Results } from '../../kit/Results.tsx'
import { StoryPage } from '../../kit/StoryPage.tsx'
import { useStoryRun } from '../../kit/useStoryRun.ts'
import { storyHref } from '../../story.ts'
import { type Build, DEFAULT_BUILD, engineOf, type Groups, isOpen, REAL, type Result, type Shape } from './compute.ts'
import source from './compute.ts?raw'
import { eruptions } from './data.ts'
import { DataFigure, GroupsFigure, WeightsFigure } from './Figures.tsx'
import { readData, readResult } from './reading.ts'

// The data are a table lookup, so the page shows them before anything runs.
const DATA = eruptions()
const READ = readData(DATA.length, DATA.wait)

const SHAPES: readonly ChoiceOption<Shape>[] = [
  { value: 'round', label: 'Round', description: 'One spread for both directions, the same for every group.' },
  { value: 'diagonal', label: 'Diagonal', description: 'Its own spread in each direction, for every group.' },
  { value: 'full', label: 'Full', description: 'Its own spread in each direction, and a tilt.' },
]

const GROUPS: readonly ChoiceOption<Groups>[] = [
  { value: 'two', label: 'Two groups', description: 'A fixed number.' },
  { value: 'three', label: 'Three groups', description: 'A fixed number.' },
  {
    value: 'open6',
    label: 'Up to 6 groups',
    description: 'The data decide how many, with stick-breaking weights.',
  },
  {
    value: 'open8',
    label: 'Up to 8 groups',
    description: 'The data decide how many, with stick-breaking weights.',
  },
]

/** What the play button says to expect: measured with the worker on a quiet machine. */
const ESTIMATE: Record<string, string> = {
  'round-two': 'about 10 seconds',
  'round-three': 'about 3 seconds',
  'round-open6': 'about 3 seconds',
  'round-open8': 'about 4 seconds',
  'diagonal-two': 'about 4 seconds',
  'diagonal-three': 'about 5 seconds',
  'diagonal-open6': 'about 5 seconds',
  'diagonal-open8': 'about 5 seconds',
  'full-two': 'about 5 seconds',
  'full-three': 'about 6 seconds',
  'full-open6': 'about 7 seconds',
  'full-open8': 'about 8 seconds',
}
const estimateOf = (build: Build): string => ESTIMATE[`${build.shape}-${build.groups}`] ?? 'a few seconds'

const fixed = (v: number, digits = 1): string => v.toFixed(digits)
const percent = (v: number): string => `${Math.round(100 * v)}%`

/** 'Gaussian mixtures': the narrative. The reader's choice sits above the play button; the run and its result sit below it. */
export default function Story(): ReactNode {
  const [build, setBuild] = useState<Build>(DEFAULT_BUILD)
  const engine = engineOf(build)
  return (
    <StoryPage>
      <p>
        Old Faithful is a geyser in Yellowstone. For each of <span className="num">{READ.rows}</span> eruptions, someone
        wrote down how long it lasted and how long the wait was before it. Both are in minutes.
      </p>

      <Figure caption="Each dot is one eruption." wide>
        <DataFigure length={DATA.length} wait={DATA.wait} />
      </Figure>

      <p>
        The dots make two clumps. Short eruptions follow short waits, and long eruptions follow long waits. A mixture
        model says that each eruption comes from one of a few hidden groups. Each group is a cloud of dots with a centre
        and a spread. The model has to find the groups from the dots alone.
      </p>

      <Equation
        label={
          <>
            A mixture: group <Tex>k</Tex> has weight <Tex>w_k</Tex>, centre <Tex>{String.raw`\muvec_k`}</Tex> and spread{' '}
            <Tex>{String.raw`\Sigmamat_k`}</Tex>
          </>
        }
        reading="The density of a point x is the sum over the groups k of the weight w k times a normal density with centre mu k and covariance Sigma k."
      >
        {String.raw`p(\xvec) = \sum_{k} w_k\, \Gauss(\xvec;\, \muvec_k,\, \Sigmamat_k)`}
      </Equation>

      <h2>Build the mixture</h2>

      <p>
        You decide how the mixture is built. The first choice is the shape of each group. Before the fit, each measure
        is put on the same scale by dividing it by its own spread.
      </p>

      <Aside>
        A spread is how far values usually stray from their centre. Statisticians call it the standard deviation.
      </Aside>

      <Choice
        label="Shape of each group"
        options={SHAPES}
        value={build.shape}
        onChange={(shape) => setBuild({ ...build, shape })}
      />

      <p>
        A tilt means that the dots inside a group lean. For example, longer eruptions may go with longer waits. Only a
        full group can follow a lean.
      </p>

      <p>
        The second choice is the number of groups. You can fix it. Or you can let the data decide: the model gets room
        for 6 or 8 groups, and their weights come from stick-breaking. The first group takes a part of a stick of length
        1. The next takes a part of what is left, and so on. A group the data do not need ends up with almost nothing.
      </p>

      <Equation
        label={
          <>
            The weights by stick-breaking: <Tex>b_k</Tex> is the part of what is left that group <Tex>k</Tex> takes
          </>
        }
        reading="Each part b k is uniform between 0 and 1. The weight w k is b k times the product, over the earlier groups j, of one minus b j."
      >
        {String.raw`b_k \sim \Beta(1, 1), \qquad w_k = b_k \prod_{j < k} (1 - b_j)`}
      </Equation>

      <Choice
        label="Number of groups"
        options={GROUPS}
        value={build.groups}
        onChange={(groups) => setBuild({ ...build, groups })}
      />

      <p data-testid="engine">
        {engine === 'ep' ? (
          <>
            Expectation propagation runs this choice. The eruptions and the groups pass messages to each other until
            nothing changes. The story{' '}
            <Link to={storyHref('how-expectation-propagation-works')} className="underline underline-offset-2">
              How expectation propagation works
            </Link>{' '}
            shows the steps.
          </>
        ) : (
          <>
            This choice is run by sampling. Gibbs sampling guesses the group of every eruption, then updates the groups,
            and repeats. Expectation propagation is too slow on it, or does not settle. The answer is the average of the
            sweeps the sampler kept.
          </>
        )}
      </p>

      <p>Press play to fit your choice. If you change a choice afterwards, the result is cleared.</p>

      <Runner key={`${build.shape}-${build.groups}`} build={build} />
    </StoryPage>
  )
}

/** The play button and everything the run makes. It is made afresh (and so empty) each time a choice changes. */
function Runner(props: { readonly build: Build }): ReactNode {
  const { build } = props
  // The worker is made only when the reader presses play. The literal has to be written here for the bundler to find it.
  // The page tells it the choice first; the worker handles messages in order, so the choice arrives before the run.
  const run = useStoryRun<Result>(() => {
    const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })
    worker.postMessage({ type: 'build', build })
    return worker
  })
  return (
    <>
      <Play
        run={run}
        label="Fit the groups"
        estimate={`${estimateOf(build)}, in the background`}
        note="the page stays free"
      />
      <Results run={run}>
        {(result) => {
          const r = readResult(result, DATA.length, DATA.wait)
          const [first, second] = r.real
          const open = isOpen(result.build.groups)
          return (
            <>
              <h2>What the model found</h2>

              <p>
                {result.engine === 'ep' ? (
                  result.settled ? (
                    <>
                      Expectation propagation settled after <span className="num">{result.sweeps}</span> sweeps.
                    </>
                  ) : (
                    <>
                      Expectation propagation stopped after <span className="num">{result.sweeps}</span> sweeps without
                      settling.
                    </>
                  )
                ) : (
                  <>
                    The sampler ran <span className="num">{result.sweeps}</span> sweeps.
                  </>
                )}{' '}
                The model had room for <span className="num">{result.pieces}</span> groups. A group holds real weight
                when it holds at least {percent(REAL)} of the eruptions. <span className="num">{r.real.length}</span>{' '}
                {r.real.length === 1 ? 'group holds' : 'groups hold'} real weight.
              </p>

              <Figure
                caption="Each dot is an eruption, coloured by its most likely group. Grey dots could belong to more than one group. Each ellipse is two spreads out from the centre of its group."
                wide
              >
                <GroupsFigure length={DATA.length} wait={DATA.wait} result={result} />
              </Figure>

              {first === undefined ? null : (
                <p>
                  The biggest group holds <span className="num">{percent(first.fit.weight)}</span> of the eruptions. Its
                  centre is an eruption of <span className="num">{fixed(first.fit.centre[0])}</span> minutes after a
                  wait of <span className="num">{fixed(first.fit.centre[1], 0)}</span> minutes.
                  {second === undefined ? null : (
                    <>
                      {' '}
                      The next holds <span className="num">{percent(second.fit.weight)}</span>, centred at{' '}
                      <span className="num">{fixed(second.fit.centre[0])}</span> and{' '}
                      <span className="num">{fixed(second.fit.centre[1], 0)}</span> minutes.
                    </>
                  )}{' '}
                  <span className="num">{r.unclear}</span> of the {r.rows} eruptions are grey.
                </p>
              )}

              <Figure caption="The share of the eruptions held by each group, biggest first. Grey bars hold less than real weight.">
                <WeightsFigure result={result} />
              </Figure>

              {r.small === 0 ? (
                <p>Every group holds real weight.</p>
              ) : (
                <p>
                  The other <span className="num">{r.small}</span> {r.small === 1 ? 'group holds' : 'groups hold'}{' '}
                  <span className="num">{percent(r.smallShare)}</span> of the eruptions
                  {r.small === 1 ? '. It is' : ' between them. They are'} not drawn.
                  {open ? '' : ' You asked for this many groups, but the data hold little for the rest.'}
                </p>
              )}

              <h2>What your choices did</h2>

              {first === undefined ? null : (
                <p>
                  {result.build.shape === 'round' ? (
                    <>
                      Round groups share one spread: <span className="num">{fixed(first.fit.spread[0], 2)}</span>{' '}
                      minutes of length and <span className="num">{fixed(first.fit.spread[1])}</span> minutes of wait.
                      Every group has the same size, and none can tilt.
                    </>
                  ) : result.build.shape === 'diagonal' ? (
                    <>
                      Each diagonal group has its own spread. The biggest group spreads{' '}
                      <span className="num">{fixed(first.fit.spread[0], 2)}</span> minutes in length and{' '}
                      <span className="num">{fixed(first.fit.spread[1])}</span> in wait. Its ellipse lines up with the
                      axes, so it cannot tilt.
                    </>
                  ) : (
                    <>
                      Each full group has its own spread and its own tilt. The biggest group spreads{' '}
                      <span className="num">{fixed(first.fit.spread[0], 2)}</span> minutes in length and{' '}
                      <span className="num">{fixed(first.fit.spread[1])}</span> in wait, and its tilt is{' '}
                      <span className="num">{fixed(first.fit.tilt, 2)}</span> (a correlation, from 0 for none to 1).
                    </>
                  )}{' '}
                  The eruptions in it {Math.abs(first.lean) < 0.2 ? 'hardly lean' : 'lean'}: their length and wait have
                  a correlation of <span className="num">{fixed(first.lean, 2)}</span>.
                  {result.build.shape === 'full' ? null : ' A group of this shape cannot follow a lean.'}
                </p>
              )}

              {open ? (
                <p>Try another shape with the same room for groups, and see how many groups the data keep.</p>
              ) : (
                <p>
                  Try the other shapes with the same number of groups. Then try letting the data decide, and see how
                  many groups hold real weight.
                </p>
              )}

              <p>
                Press <em>Run again</em> to hand the eruptions to the method in a new order
                {result.engine === 'gibbs' ? ' and to start the sampler from a new seed' : ''}. This shows how much the
                answer moves with the seed.
              </p>

              <Code source={source} region={result.engine} title="compute.ts" />
            </>
          )
        }}
      </Results>
    </>
  )
}
