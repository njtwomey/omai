import type { ReactNode } from 'react'
import { Aside, Code, Figure } from '../../kit/Blocks.tsx'
import { Tex, tex } from '../../kit/Maths.tsx'
import { Play } from '../../kit/Play.tsx'
import { Results } from '../../kit/Results.tsx'
import { StoryPage } from '../../kit/StoryPage.tsx'
import { useStoryRun } from '../../kit/useStoryRun.ts'
import type { Result } from './compute.ts'
import source from './compute.ts?raw'
import { simulate } from './data.ts'
import { AverageFigure, CutsFigure, DataFigure, GreedyPanels } from './Figures.tsx'
import { readResult } from './reading.ts'

// The data are sixty fixed points, drawn here so the page shows them before anything runs.
const DATA = simulate()

const fixed = (v: number, digits = 2): string => v.toFixed(digits)
const percent = (v: number): string => `${Math.round(v * 100)}%`

/** 'Trees, one split at a time, and all trees at once': the narrative, with the play button where the computation starts. */
export default function Story(): ReactNode {
  // The worker is made only when the reader presses play. The literal has to be written here for the bundler to find it.
  const run = useStoryRun<Result>(() => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }))
  return (
    <StoryPage>
      <p>
        Here are sixty measurements. For each one, we set a value <Tex>x</Tex> and read off a value <Tex>y</Tex>. The
        points rise from left to right. The noise is large, so it is hard to say where the level changes.
      </p>

      <Figure caption="The sixty points. Look for the places where the level changes." wide>
        <DataFigure data={DATA} />
      </Figure>

      <p>
        A decision tree describes data like this with cuts. A cut at one value of <Tex>x</Tex> splits the points into
        two groups, and each group is a <em>leaf</em>. Each leaf keeps an estimate of the average <Tex>y</Tex> of its
        points, and how sure it is of that estimate. A leaf with more points is more sure.
      </p>

      <h2>Which cuts to make</h2>

      <p>
        The first method grows one tree. It tries every cut and scores each by how much more likely it makes the data.
        It takes the best cut, then looks again inside each new leaf. It stops when no cut is worth its cost. Each extra
        cut costs something, because the tree prior says that simple trees are more likely.
      </p>

      <p>
        The second method does not pick one tree. It gives every tree of up to two cuts a weight, from how likely the
        data are under that tree. These weights are the <em>posterior</em>: the probability of each tree, given the
        data. Then it draws trees from the weights, and averages their predictions.
      </p>

      <Play run={run} label="Grow the trees" estimate="about a second" />

      <Results run={run}>
        {(result) => {
          const r = readResult(result)
          const first = result.greedy.cuts[0]
          const second = result.greedy.cuts[1]
          const next = result.greedy.next
          const afterOne = result.greedy.stages[1]?.leaves ?? []
          return (
            <>
              <h2>One tree, one split at a time</h2>

              <p>
                The first cut is at <Tex>{tex`x = ${fixed(r.firstCut)}`}</Tex>. Its score is{' '}
                <span className="num">{fixed(first?.gain ?? 0, 1)}</span>, the log of how much more likely the data are
                with the cut than without. It splits the <span className="num">{first?.n}</span> points into{' '}
                <span className="num">{first?.nLeft}</span> and <span className="num">{first?.nRight}</span>.
                {second === undefined ? null : (
                  <>
                    {' '}
                    The second cut is at <Tex>{tex`x = ${fixed(r.secondCut)}`}</Tex>. Its score is{' '}
                    <span className="num">{fixed(second.gain, 1)}</span>. After the cost of the extra cut, its odds are{' '}
                    <span className="num">{fixed(second.logOdds, 1)}</span> on a log scale. That is above zero, so the
                    cut is made.
                  </>
                )}
              </p>

              <Figure
                caption="The tree with no cut and after each cut. Solid: the mean of each leaf. Shaded: two standard deviations of that mean. Dashed: the cuts."
                wide
              >
                <GreedyPanels data={DATA} result={result} />
              </Figure>

              <p>
                {next === undefined ? (
                  'No cut was left to try.'
                ) : (
                  <>
                    The best cut left is at <Tex>{tex`x = ${fixed(next.threshold)}`}</Tex>. Its score is{' '}
                    <span className="num">{fixed(next.gain, 1)}</span>, and its odds are{' '}
                    <span className="num">{fixed(next.logOdds, 1)}</span>. It does not pay, so the tree stops.
                  </>
                )}{' '}
                Each leaf also has a step size, one over its count plus one: <Tex>1 / (n + 1)</Tex>. It is how far one
                new point would move the leaf's mean. After the first cut, a leaf of{' '}
                <span className="num">{afterOne[0]?.n}</span> points has a step of{' '}
                <span className="num">{fixed(afterOne[0]?.stepSize ?? 0, 3)}</span>, and a leaf of{' '}
                <span className="num">{afterOne[1]?.n}</span> points has{' '}
                <span className="num">{fixed(afterOne[1]?.stepSize ?? 0, 3)}</span>. A leaf with few points keeps a
                large step, and a leaf with many points takes small ones.
              </p>

              <h2>All the trees at once</h2>

              <p>
                The posterior covers <span className="num">{result.posterior.trees.toLocaleString('en-GB')}</span>{' '}
                trees. Each uses at most two of <span className="num">{result.posterior.thresholds.length}</span>{' '}
                candidate cuts. The weights are exact. They come from formulas, with no sampling. The posterior gives{' '}
                <span className="num">{percent(r.one)}</span> to trees with one cut,{' '}
                <span className="num">{percent(r.two)}</span> to two cuts and{' '}
                <span className="num">{percent(r.three)}</span> to three.
              </p>

              <Figure
                caption="The average over all trees (solid) with its band, thirty trees drawn from the posterior (thin), and the greedy tree (dashed)."
                wide
              >
                <AverageFigure data={DATA} result={result} />
              </Figure>

              <p>
                The average and the greedy tree differ most near <Tex>{tex`x = ${fixed(r.widestAt, 1)}`}</Tex>, by{' '}
                <span className="num">{fixed(r.widestDifference)}</span>. The greedy tree jumps at one place. The
                average rises over a stretch, because the trees disagree about where the cuts are.
              </p>

              <Aside>
                The thin trees come from a sampler. It starts from a tree with no cuts and makes random moves: add a
                cut, remove one, or change one. It keeps a move with a probability set by the posterior.{' '}
                {result.sampled.chains} chains made{' '}
                <span className="num">{result.sampled.draws.toLocaleString('en-GB')}</span> draws, and{' '}
                <span className="num">{percent(result.sampled.acceptance)}</span> of the moves were kept.
              </Aside>

              <h2>Where the cuts could be</h2>

              <Figure
                caption="The chance that a cut is at each position, in any tree. Dashed: the greedy tree's cuts. Dotted: the cuts the data were made with."
                wide
              >
                <CutsFigure result={result} />
              </Figure>

              <p>
                The first cut is settled. A cut lies within 0.3 of <Tex>{tex`x = ${fixed(r.firstCut)}`}</Tex> with a
                chance of <span className="num">{percent(Math.min(1, r.nearFirst))}</span>. The second is not. Positions
                from <span className="num">{fixed(r.secondSpan?.from ?? 0, 1)}</span> to{' '}
                <span className="num">{fixed(r.secondSpan?.to ?? 0, 1)}</span> each have a chance of at least 5%, and
                the greedy position has <span className="num">{percent(Math.min(1, r.nearSecond))}</span>. The single
                most probable tree has cuts at{' '}
                <span className="num">{r.top.cuts.map((c) => fixed(c)).join(' and ')}</span>, and it holds only{' '}
                <span className="num">{percent(r.top.probability)}</span> of the posterior.
              </p>

              <p>
                The greedy tree is quick to grow and easy to read. It gives one tree, and nothing about the others. The
                posterior shows how many trees fit, and where they differ. The sampler gives nearly the same answer as
                the exact weights: its average curve is within{' '}
                <span className="num">{fixed(result.sampled.curveGap, 3)}</span> of the exact one. It also works for
                trees too large to list.
              </p>

              <Code source={source} title="compute.ts" />
            </>
          )
        }}
      </Results>
    </StoryPage>
  )
}
