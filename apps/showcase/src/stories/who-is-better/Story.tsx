import type { ReactNode } from 'react'
import { Link } from '../../../../site/src/shell/Link.tsx'
import { Aside, Code, Figure } from '../../kit/Blocks.tsx'
import { Play } from '../../kit/Play.tsx'
import { Results } from '../../kit/Results.tsx'
import { StoryPage } from '../../kit/StoryPage.tsx'
import { useStoryRun } from '../../kit/useStoryRun.ts'
import { storyHref } from '../../story.ts'
import type { Result } from './compute.ts'
import source from './compute.ts?raw'
import { DRAW_MARGIN, EARLY, PLAYERS, ROUNDS, tournament } from './data.ts'
import { LadderFigure, TrajectoryFigure } from './Figures.tsx'
import { readResult, tally } from './reading.ts'

// The games are made from a fixed seed, so the page shows their results before anything runs.
const GAMES = tournament()
const RECORDS = tally(GAMES)
const DRAWS = GAMES.result.filter((r) => r === 1).length

const fixed = (v: number, digits = 2): string => v.toFixed(digits)
const signed = (v: number): string => `${v < 0 ? '-' : '+'}${Math.abs(v).toFixed(2)}`
const names = (list: readonly string[]): string =>
  list.length <= 1 ? (list[0] ?? '') : `${list.slice(0, -1).join(', ')} and ${list.at(-1)}`

/** The record of every player, as a small table. */
function Scoreboard(): ReactNode {
  return (
    <div className="story-block overflow-x-auto rounded-lg border border-line bg-surface p-3 font-sans text-sm">
      <table className="w-full border-collapse text-left tabular-nums">
        <thead>
          <tr className="text-muted">
            <th className="py-1 pr-4 font-medium">Player</th>
            <th className="py-1 pr-4 font-medium">Won</th>
            <th className="py-1 pr-4 font-medium">Drawn</th>
            <th className="py-1 font-medium">Lost</th>
          </tr>
        </thead>
        <tbody>
          {RECORDS.map((r) => (
            <tr key={r.name} className="border-t border-line">
              <td className="py-1 pr-4">{r.name}</td>
              <td className="py-1 pr-4">{r.wins}</td>
              <td className="py-1 pr-4">{r.draws}</td>
              <td className="py-1">{r.losses}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/** 'Who is better?': the narrative, with the play button where the message passing starts. */
export default function Story(): ReactNode {
  // The worker is made only when the reader presses play. The literal has to be written here for the bundler to find it.
  const run = useStoryRun<Result>(() => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }))
  const mostWins = [...RECORDS].sort((a, b) => b.wins - a.wins)[0]
  const fewestWins = [...RECORDS].sort((a, b) => a.wins - b.wins)[0]
  return (
    <StoryPage>
      <p>
        Five players play each other. Every pair meets {ROUNDS} times, so there are{' '}
        <span className="num">{GAMES.a.length}</span> games, and each game ends in a win, a loss or a draw (
        <span className="num">{DRAWS}</span> of them are draws). Nobody has said how good any player is. All we have is
        the results.
      </p>

      <Scoreboard />

      <p>
        We can count. {mostWins?.name} won the most games (<span className="num">{mostWins?.wins}</span>) and{' '}
        {fewestWins?.name} the fewest (<span className="num">{fewestWins?.wins}</span>). A count leaves two things out.
        It does not say how much better one player is than another. It also does not say how sure we can be, when each
        player has played only <span className="num">{(PLAYERS.length - 1) * ROUNDS}</span> games. A rating can say
        both.
      </p>

      <h2>What a rating is</h2>

      <p>
        We give every player a skill. It is a number that we cannot see, and a bigger number means a better player. We
        also say how sure we are about it, by describing it as a bell curve. The centre of the bell curve is the best
        guess, which is called the mean. How wide it is says how unsure we are, and that width is called the spread.
        Before any game we know nothing, so every player starts with the same bell curve: a mean of 0 and a spread of 1.
      </p>

      <Aside>
        A bell curve is a normal distribution. It is fully described by its mean and its spread (the standard
        deviation), which is why a message in this story is just two numbers.
      </Aside>

      <p>
        Next we say how a game is played. In each game, each player performs a little better or worse than their skill,
        by chance. The better performance wins. If the two performances are within {DRAW_MARGIN} of each other, the game
        is a draw. So the result of a game tells us only about the difference between the two skills.
      </p>

      <h2>How the messages work</h2>

      <p>
        The method is called expectation propagation. The model is a graph. Each player is a node, and each game is a
        factor that joins the two players of that game. The method sends messages along the graph. A message is a bell
        curve, with a mean and a spread, sent from one part of the model to another. It runs in these steps.
      </p>

      <ol className="mb-5 list-decimal">
        <li>Every player starts with the bell curve for their skill: mean 0, spread 1.</li>
        <li>
          A game takes the current bell curves of its two players and looks at its result. If the first player won, the
          first skill is probably higher than the second. The game sends each of its two players a message: a new bell
          curve for that player, which says what this one result suggests.
        </li>
        <li>
          Every player multiplies the messages from all their games with their starting bell curve. The product is the
          player&rsquo;s new bell curve.
        </li>
        <li>
          The games send messages again. This time a game removes its own earlier message from each player&rsquo;s bell
          curve first, so that a result is not counted twice.
        </li>
        <li>
          One pass over all the games, and then all the players, is a sweep. The sweeps repeat until no message changes.
        </li>
      </ol>

      <p>
        There is one approximation. The exact effect of a win is not a bell curve. It cuts off the part of the picture
        where the winner has the lower skill. Each game therefore sends the bell curve that has the same mean and the
        same spread as that cut-off shape. This is the step that gives the method its name: it matches the expected
        values, the mean and the spread, and passes them on.
      </p>

      <Play run={run} label="Rate the players" estimate="about 6 seconds, in the background" />

      <Results run={run}>
        {(result) => {
          const r = readResult(result)
          return (
            <>
              <h2>The ratings after {r.games} games</h2>

              <p>
                The messages stopped changing after <span className="num">{result.sweeps}</span> sweeps. The figure
                shows the bell curve of every player, the best at the top. The diamond under each bell is the skill that
                player was given when the games were made up. The method never saw it.
              </p>

              <Figure
                caption="The rating of each player after all the games: a bell curve with the best guess at its peak. The width is how unsure the rating is. The diamond is the skill the games were made from."
                wide
              >
                <LadderFigure result={result} />
              </Figure>

              <p>
                {r.ladder[0]?.name} has the highest rating, <span className="num">{fixed(r.ladder[0]?.mean ?? 0)}</span>
                , and {r.ladder.at(-1)?.name} the lowest,{' '}
                <span className="num">{fixed(r.ladder.at(-1)?.mean ?? 0)}</span>. Every spread is about{' '}
                <span className="num">{fixed(r.ladder[0]?.sd ?? 0)}</span>, which is what we expect, because every
                player played the same number of games.{' '}
                {r.inTrueOrder
                  ? 'The order of the ratings is the order of the true skills. '
                  : 'The order of the ratings is not quite the order of the true skills. '}
                Of the five true skills, <span className="num">{r.covered}</span> lie inside the 90% range of their
                player&rsquo;s bell curve.
              </p>

              {r.overlapping.length === 0 ? (
                <p>
                  No two neighbours on the ladder have overlapping 90% ranges, so these games are enough to put the five
                  players in order.
                </p>
              ) : (
                <p>
                  The 90% ranges of {names(r.overlapping.map(([a, b]) => `${a} and ${b}`))} overlap. For each of those
                  pairs, the games so far leave some doubt about which player is better.
                </p>
              )}

              <h2>How the ratings got there</h2>

              <p>
                The run also rated the players after the first game, after the second, and so on up to game {EARLY}, and
                then after every five games. The games arrive in a random order, which changes each time you press{' '}
                <em>Run again</em>; the final ratings do not change.
              </p>

              <Figure
                caption="The rating of each player after each number of games. The line is the mean and the band is one spread on each side. At zero games every player has the same bell curve."
                wide
              >
                <TrajectoryFigure result={result} />
              </Figure>

              <p>
                At zero games every spread is <span className="num">1.00</span>. After 10 games the average spread is{' '}
                <span className="num">{fixed(r.spreadAfter(10) ?? 0)}</span>, and after {r.games} games it is{' '}
                <span className="num">{fixed(r.spreadAfter(r.games) ?? 0)}</span>. Each game makes the two players it
                involves a little more certain, and it moves their means.
              </p>

              {r.upset === undefined || r.expected === undefined || r.upset.game === r.expected.game ? null : (
                <>
                  <p>
                    Two of the first games show how much a result can move a rating. In game{' '}
                    <span className="num">{r.upset.game}</span>, {r.upset.winner} beat {r.upset.loser}. Before that
                    game, {r.upset.loser}&rsquo;s mean was <span className="num">{signed(r.upset.gap)}</span> from{' '}
                    {r.upset.winner}&rsquo;s (positive means {r.upset.loser} was ahead), and the win raised{' '}
                    {r.upset.winner}
                    &rsquo;s mean by <span className="num">{fixed(r.upset.rise)}</span>. In game{' '}
                    <span className="num">{r.expected.game}</span>, {r.expected.winner} beat {r.expected.loser}, whose
                    mean was <span className="num">{signed(r.expected.gap)}</span> from {r.expected.winner}&rsquo;s, and
                    the win raised {r.expected.winner}&rsquo;s mean by{' '}
                    <span className="num">{fixed(r.expected.rise)}</span>.
                  </p>
                  <p>
                    {r.upsetMovedMore
                      ? 'In these two games, the win over the higher-rated player moved the rating more than the win over the lower-rated one. This is what we expect. A result that the ratings had already predicted carries little news, and a result that they had not predicted carries more.'
                      : 'In these two games the win over the higher-rated player did not move the rating more. The early ratings are still wide, and the means move by different amounts for several reasons, so two games do not settle it.'}
                  </p>
                </>
              )}

              <h2>Why this is useful</h2>

              <p>
                The same problem appears whenever results come in pairs: chess ratings, matching players in an online
                game, comparing two models on a set of tests. The messages are what make it scale. A game only talks to
                its two players, so a new result needs one more message and not a new calculation from the start.
              </p>

              <p>
                The ratings are approximate, because each message is a bell curve fitted to a cut-off shape. Two other
                stories use the same messages on other problems:{' '}
                <Link to={storyHref('two-humps')} className="underline underline-offset-2">
                  Two humps
                </Link>{' '}
                and{' '}
                <Link to={storyHref('when-did-the-rate-change')} className="underline underline-offset-2">
                  When did the rate change?
                </Link>
                .
              </p>
              <Code source={source} title="compute.ts" />
            </>
          )
        }}
      </Results>
    </StoryPage>
  )
}
