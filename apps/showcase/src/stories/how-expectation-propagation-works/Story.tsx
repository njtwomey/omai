import type { ReactNode } from 'react'
import { Link } from '../../../../site/src/shell/Link.tsx'
import { Code, Figure, Reveal } from '../../kit/Blocks.tsx'
import { Play } from '../../kit/Play.tsx'
import { StoryPage } from '../../kit/StoryPage.tsx'
import { useStoryRun } from '../../kit/useStoryRun.ts'
import { storyHref } from '../../story.ts'
import type { Result } from './compute.ts'
import source from './compute.ts?raw'
import { CLUTTER_CHANCE, CLUTTER_SD, MAX_SWEEPS, PRIOR_SD, ROWS, SHOWN_SWEEP, TOLERANCE } from './data.ts'
import { AnswerFigure, ChangeFigure, GraphArt, SweepsFigure, UpdateFigure } from './Figures.tsx'
import { bellText, clutterPositions, fixed, listOf, messageText, readExact, readSweeps, sci } from './reading.ts'

const FIRST_SEED = 37

const linkClass = 'underline underline-offset-2'

/** A line of formula, set apart from the prose. */
function Formula(props: { readonly children: ReactNode }): ReactNode {
  return (
    <div className="my-3 overflow-x-auto whitespace-pre rounded-md border border-line bg-surface px-4 py-2.5 font-mono text-[0.9rem] leading-relaxed">
      {props.children}
    </div>
  )
}

/** A small table in the page's sans-serif, scrolling sideways on a narrow screen. */
function Table(props: {
  readonly head: readonly string[]
  readonly rows: readonly (readonly ReactNode[])[]
}): ReactNode {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse font-sans text-[0.9rem] tabular-nums">
        <thead>
          <tr>
            {props.head.map((h) => (
              <th key={h} className="border-b border-line px-2.5 py-1.5 text-left font-semibold text-muted">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {props.rows.map((row, i) => (
            // biome-ignore lint/suspicious/noArrayIndexKey: the rows are a fixed list, never reordered
            <tr key={i} className="border-b border-line last:border-0">
              {row.map((cell, j) => (
                // biome-ignore lint/suspicious/noArrayIndexKey: the cells are a fixed list, never reordered
                <td key={j} className="px-2.5 py-1.5 align-top">
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/** 'How expectation propagation works': the narrative, with the play button where the first computed figure is needed. */
export default function Story(): ReactNode {
  // The worker is made only when the reader presses play. The literal has to be written here for the bundler to find it.
  const run = useStoryRun<Result>(
    () => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }),
    FIRST_SEED,
  )
  const result = run.result
  return (
    <StoryPage>
      <p>
        Expectation propagation, or EP, is a way to approximate the posterior of a model when the posterior cannot be
        written down. This post follows one run of omai&rsquo;s EP on one small model, from the model to the answer, in
        the order things happen. After the play button, every number comes from that run. When you have read it you
        should be able to say what EP does to each factor of a model, and to read an EP run in omai: its messages, its
        sweeps and the point where it stops.
      </p>

      <h2>1. The problem</h2>

      <p>
        We start with a model and some data. The model says how the data could have come from some unknown numbers. What
        we want is the posterior: what we believe about the unknowns once the data are taken into account. By
        Bayes&rsquo; rule, the posterior is the prior (what we believed before) times the likelihood of each data point,
        divided by one constant that makes the total probability equal one.
      </p>

      <p>
        That product is the difficulty. For a few models the product of the factors has a known shape, and the posterior
        is a formula. For most models it does not. The factors have different shapes, their product has no name, and the
        constant is an integral with no closed form. With one unknown we can still compute the posterior numerically,
        and we will, to check the result. With tens or thousands of unknowns we cannot.
      </p>

      <p>
        So we settle for an approximation of a form we can handle. EP uses the Gaussian, the bell curve. A bell is
        described by two numbers: its mean, where it is centred, and its variance, how wide it is. The standard
        deviation, written sd, is the square root of the variance. EP&rsquo;s answer is one bell for each unknown: its
        mean and its spread are the belief.
      </p>

      <p>
        The model used from here to the end is Minka&rsquo;s clutter problem, the example in the paper that introduced
        EP (2001). There is one unknown, a location, written <code>θ</code>. We have {ROWS} measurements. With
        probability {fixed(1 - CLUTTER_CHANCE, 1)}, a measurement is a reading of the location: a draw from a bell of sd
        1 around <code>θ</code>. With probability {fixed(CLUTTER_CHANCE, 1)} it is clutter: a draw from a much wider
        bell, with sd {fixed(CLUTTER_SD, 1)}, centred on zero and unrelated to <code>θ</code>. We are not told which
        measurements are clutter. Before seeing them we believe <code>θ</code> is within about {PRIOR_SD} of zero: a
        bell with mean 0 and sd {PRIOR_SD}.
      </p>

      <Formula>
        posterior(θ) ∝ N(θ; 0, {PRIOR_SD}²) × ∏ᵢ [ {fixed(1 - CLUTTER_CHANCE, 1)} · N(yᵢ; θ, 1²) +{' '}
        {fixed(CLUTTER_CHANCE, 1)} · N(yᵢ; 0, {fixed(CLUTTER_SD, 1)}²) ]
      </Formula>

      <p>
        Each of the {ROWS} brackets is a sum of two bells, so multiplying them out gives up to 2<sup>{ROWS}</sup> ={' '}
        {2 ** ROWS} bell-shaped terms, one for each way of deciding which measurements are clutter. With one unknown we
        can evaluate the whole product on a fine grid of values of <code>θ</code> and get the exact posterior. We do
        that at the end. EP does not use the grid, and its work grows with the number of factors, not with the{' '}
        {2 ** ROWS}.
      </p>

      <h2>2. The model as a graph of factors</h2>

      <p>
        Each part of the formula is a factor: a function of the unknowns that the posterior is multiplied by. There is
        one for the prior and one for each measurement, so {ROWS + 1} in all. A factor graph draws them: a circle for
        each unknown, a square for each factor, and a line from each factor to the unknowns it mentions.
      </p>

      <Figure caption="The factor graph of the model. One unknown, and one factor for the prior and for each of the six measurements. The factor f0 is the prior; f1 to f6 are the measurements.">
        <GraphArt />
      </Figure>

      <p>
        Here there is one unknown, so every factor joins the same circle. In a larger model a factor joins only the few
        unknowns it mentions, and each unknown only hears from the factors around it. That is what keeps EP cheap on big
        models. The one-unknown case shows every step of the method except the passing of messages between different
        unknowns.
      </p>

      <h2>3. The idea: a stand-in for each factor</h2>

      <p>
        The product of the factors is hard because the factors are not bells. EP replaces each factor by a bell, a
        stand-in, so that the product of the stand-ins is easy. A product of bells is again a bell, so the approximation
        of the posterior is the product of all the stand-ins. Call that product <code>q</code>.
      </p>

      <p>
        A stand-in sent from a factor to an unknown is called a message. A message is a bell, up to a constant that is
        never tracked. There are two ways to write a bell. One is the mean and the variance. The one the engine uses is
        the precision, which is one over the variance, and the precision times the mean. Call them <code>J</code> and{' '}
        <code>h</code>; together they are the natural parameters of the bell.
      </p>

      <p>
        They are used because multiplying two bells then means adding. If two messages have <code>(h₁, J₁)</code> and{' '}
        <code>(h₂, J₂)</code>, their product has <code>(h₁ + h₂, J₁ + J₂)</code>. Dividing one bell by another means
        subtracting. The mean of a message is <code>h / J</code> and its variance is <code>1 / J</code>. For example,
        the prior has mean 0 and sd {PRIOR_SD}, so <code>J = {1 / PRIOR_SD ** 2}</code> and <code>h = 0</code>. A bell
        with mean 2 and sd 1 has <code>J = 1</code> and <code>h = 2</code>. Their product has <code>J = 1.01</code> and{' '}
        <code>h = 2</code>, which is a bell with mean 1.98 and sd 0.995.
      </p>

      <p>
        Two more facts about messages. A message that says nothing has <code>J = 0</code> and <code>h = 0</code>; call
        it flat. And a message may have a negative <code>J</code>. It is then not a bell, only a pair of numbers that
        still add and subtract correctly. That is allowed as long as the sum of all the messages, the approximation{' '}
        <code>q</code> itself, has a positive <code>J</code>.
      </p>

      <p>
        EP starts with every stand-in flat, except the prior&rsquo;s, which is already a bell and is used as it is. The
        task is to choose the other {ROWS} stand-ins. A good stand-in does not have to match its factor everywhere. It
        has to match where the posterior has its mass, and that depends on the other factors. So EP cannot choose each
        stand-in alone. It chooses one at a time, with the others held fixed, and repeats. One such choice is called an
        update.
      </p>

      <h2>4. One update, step by step</h2>

      <p>
        An update has five steps. They are written here for a factor <code>k</code> with real factor <code>f_k</code>{' '}
        and current stand-in <code>g_k</code>. The current approximation is <code>q</code>, the product of all the
        stand-ins.
      </p>

      <ol>
        <li>
          Take the factor&rsquo;s own stand-in out. Divide <code>q</code> by <code>g_k</code>. The result is the cavity:
          what we believe about <code>θ</code> from everything except factor <code>k</code>. In natural parameters,
          subtract the message.
          <Formula>{'cavity(θ) = q(θ) / g_k(θ)        (h, J) = (h_q − h_k, J_q − J_k)'}</Formula>
        </li>
        <li>
          Multiply the cavity by the real factor, not its stand-in. The result is the tilted distribution: what we would
          believe if factor <code>k</code> were exact and the others still approximate. In general it is not a bell,
          because <code>f_k</code> is not one.
          <Formula>tilted(θ) ∝ cavity(θ) × f_k(θ)</Formula>
        </li>
        <li>
          Find the bell with the same mean and variance as the tilted distribution. This is moment matching. Of all
          bells, it is the one closest to the tilted distribution in the sense of the Kullback&ndash;Leibler divergence
          from the tilted distribution to the bell. It needs two integrals, the mean and the variance of the tilted
          distribution. This is the only step that uses the real factor, and the only one that can be hard. The library
          has closed forms for the factors that allow them and numerical integration for the rest; the clutter factor
          here is a numerical one.
          <Formula>
            {'m = ∫ θ · tilted(θ) dθ        v = ∫ (θ − m)² · tilted(θ) dθ        J = 1 / v,  h = m / v'}
          </Formula>
        </li>
        <li>
          Divide the matched bell by the cavity. The result is the new stand-in. In natural parameters, subtract the
          cavity from the matched bell.
          <Formula>{'g_k(new) = matched(θ) / cavity(θ)        (h, J) = (h_m − h_cavity, J_m − J_cavity)'}</Formula>
        </li>
        <li>
          Replace the old stand-in by the new one. The approximation is now the cavity times the new stand-in, which is
          the matched bell. After the update, <code>q</code> has the same mean and variance as the tilted distribution
          of factor <code>k</code>.
        </li>
      </ol>

      <p>
        To see these steps with real numbers we need a run. Press play once. In a background worker the page draws{' '}
        {ROWS} measurements from the seed, runs omai&rsquo;s EP on the model above one factor at a time until it
        settles, then takes one update apart with tensor operations and compares it with the message the engine itself
        sent. Everything below is read from that one run.
      </p>

      <Play run={run} label="Run expectation propagation" estimate="about a second" />

      {result === undefined ? (
        <p className="text-muted">
          Nothing has been computed yet. When you press play, the measurements, one update with its numbers, the sweeps,
          the worked example and the comparison with the exact answer appear below, in the order of the post.
        </p>
      ) : (
        <Reveal>
          <RunStory result={result} />
        </Reveal>
      )}

      <Code source={source} title="compute.ts" />
    </StoryPage>
  )
}

/** Everything that reads the result, from the first update to the closing pointers. */
function RunStory(props: { readonly result: Result }): ReactNode {
  const { result } = props
  const taken = result.taken
  const sweeps = readSweeps(result)
  const exact = readExact(result)
  const clutter = clutterPositions(result)
  const cavityMean = taken === undefined ? Number.NaN : taken.cavity.shift / taken.cavity.precision
  const cavitySd = taken === undefined ? Number.NaN : 1 / Math.sqrt(taken.cavity.precision)
  const beforeBell =
    taken === undefined
      ? null
      : { mean: taken.before.shift / taken.before.precision, sd: 1 / Math.sqrt(taken.before.precision) }
  const sweepTable = result.standIns
  const beliefAfter = (s: number): string =>
    bellText({ mean: result.beliefs.mean[s - 1] as number, sd: result.beliefs.sd[s - 1] as number })
  const firstUpdates = result.first
  const second = firstUpdates[1]
  const lastUpdate = firstUpdates.at(-1)
  return (
    <>
      <h2>4, continued: one update with numbers</h2>

      <p>
        The run drew {result.rows} measurements: {listOf(result.points.map((p) => fixed(p, 2)))}. The generator also
        kept which of them were clutter
        {clutter.length === 0 ? (
          <>: none were, this time.</>
        ) : (
          <>
            : {clutter.length === 1 ? 'measurement' : 'measurements'} {listOf(clutter)}.
          </>
        )}{' '}
        EP is not told this. It sees only the values.
      </p>

      {taken === undefined ? (
        <p>
          In this run no factor had a usable cavity in sweep {SHOWN_SWEEP}, so there is no update to take apart here.
          Press <em>Run again</em> for another set of measurements.
        </p>
      ) : (
        <>
          <p>
            We look at the update of sweep {taken.sweep}, because by then every factor has a stand-in of its own and
            step 1 has something to take out. We take the measurement whose tilted curve is furthest from a bell, so
            that the difference can be seen. That is measurement <span className="num">{taken.factor}</span>, with value{' '}
            <span className="num">{fixed(taken.point)}</span>.
          </p>

          <Figure
            wide
            caption={`One update, sweep ${taken.sweep}. Blue: the cavity. Dotted: the real factor of this measurement as a function of the location, scaled to fit. Orange: the tilted curve, the product of the two. Dashed: the bell with the same mean and variance as the orange curve, which becomes the new approximation.`}
          >
            <UpdateFigure taken={taken} />
          </Figure>

          <p>
            Before the update the approximation had mean{' '}
            <span className="num">{fixed(beforeBell?.mean ?? Number.NaN)}</span> and sd{' '}
            <span className="num">{fixed(beforeBell?.sd ?? Number.NaN)}</span>. Step 1 removes this factor&rsquo;s own
            stand-in, which had precision <span className="num">{fixed(taken.own.precision)}</span>. What is left, the
            cavity, has mean <span className="num">{fixed(cavityMean)}</span> and sd{' '}
            <span className="num">{fixed(cavitySd)}</span>.
          </p>

          <p>
            Step 2 multiplies the cavity by the real factor. The real factor is the dotted curve: a low floor from the
            clutter explanation and a bump around the measurement from the reading explanation. The product, the orange
            curve, has mean <span className="num">{fixed(taken.tilted.mean)}</span> and sd{' '}
            <span className="num">{fixed(taken.tilted.sd)}</span>. Its mean is{' '}
            <span className="num">{fixed(Math.abs(taken.tilted.mean - cavityMean))}</span>{' '}
            {taken.tilted.mean > cavityMean ? 'above' : 'below'} the cavity&rsquo;s, and its sd is{' '}
            {taken.tilted.sd < cavitySd ? 'smaller' : 'larger'}. It is not a bell, because the real factor is not one.
          </p>

          <p>
            Step 3 is the dashed bell. It has the mean and sd of the orange curve, so{' '}
            <span className="num">{fixed(taken.tilted.mean)}</span> and{' '}
            <span className="num">{fixed(taken.tilted.sd)}</span>, which in natural parameters is{' '}
            <code>h = {fixed(taken.matched.shift)}</code> and <code>J = {fixed(taken.matched.precision)}</code>. The
            divergence from the orange curve to this bell is <span className="num">{fixed(taken.divergence, 3)}</span>{' '}
            nats; it is zero only if the tilted curve is a bell. Step 4 subtracts the cavity from it: the new stand-in
            has <code>h = {fixed(taken.standIn.shift)}</code> and <code>J = {fixed(taken.standIn.precision)}</code>
            {taken.standIn.precision > 0
              ? '.'
              : ', a negative precision. The matched bell is wider than the cavity, so the new stand-in has to widen it. It is not a bell, only a correction, and it is allowed because its sum with the cavity is a bell.'}{' '}
            Step 5 puts it in place of the old one. The approximation after the update is the dashed bell.
          </p>

          <Figure caption="The same update as numbers. Precision is J, one over the variance. For the tilted curve only the mean and sd are listed, because it is not a bell.">
            <Table
              head={['', 'mean', 'sd', 'precision J', 'h']}
              rows={[
                [
                  'approximation before',
                  fixed(beforeBell?.mean ?? Number.NaN),
                  fixed(beforeBell?.sd ?? Number.NaN),
                  fixed(taken.before.precision),
                  fixed(taken.before.shift),
                ],
                [
                  'old stand-in',
                  taken.own.precision > 0 ? fixed(taken.own.shift / taken.own.precision) : '—',
                  taken.own.precision > 0 ? fixed(1 / Math.sqrt(taken.own.precision)) : '—',
                  fixed(taken.own.precision),
                  fixed(taken.own.shift),
                ],
                [
                  '1. cavity',
                  fixed(cavityMean),
                  fixed(cavitySd),
                  fixed(taken.cavity.precision),
                  fixed(taken.cavity.shift),
                ],
                ['2. tilted (not a bell)', fixed(taken.tilted.mean), fixed(taken.tilted.sd), '—', '—'],
                [
                  '3. matched bell',
                  fixed(taken.tilted.mean),
                  fixed(taken.tilted.sd),
                  fixed(taken.matched.precision),
                  fixed(taken.matched.shift),
                ],
                [
                  '4. new stand-in',
                  taken.standIn.precision > 0 ? fixed(taken.standIn.shift / taken.standIn.precision) : '—',
                  taken.standIn.precision > 0 ? fixed(1 / Math.sqrt(taken.standIn.precision)) : '—',
                  fixed(taken.standIn.precision),
                  fixed(taken.standIn.shift),
                ],
              ]}
            />
          </Figure>

          <p>
            The page worked this out itself, with tensor operations on a grid of 401 values of the location (the code is
            at the end). The engine does the same update with its own numerical integration and sends a message. The two
            stand-ins agree: the largest difference between their two numbers is within{' '}
            <span className="num">{sci(Math.max(taken.engineGap, 1e-12))}</span>.
          </p>
        </>
      )}

      <h2>5. The sweep</h2>

      <p>
        EP does this update for every factor in turn: the prior first, then measurements 1 to {ROWS}. Going once through
        all {ROWS + 1} is a sweep. After a sweep every stand-in has changed, so the cavities the early factors used are
        out of date. EP sweeps again, and again. In omai this is the engine behind <code>numericEpMethod</code>; the
        setting <code>granularity: &apos;factor&apos;</code> makes each step of the run one update, so a sweep is{' '}
        {ROWS + 1} steps.
      </p>

      <Figure
        wide
        caption="The belief about the location after each sweep: the mean as a line and one sd either side as a band. The dashed line is the exact mean from the grid."
      >
        <SweepsFigure result={result} />
      </Figure>

      <p>
        After sweep 1 the belief had mean{' '}
        <span className="num">{sweeps.first === null ? '—' : fixed(sweeps.first.mean)}</span> and sd{' '}
        <span className="num">{sweeps.first === null ? '—' : fixed(sweeps.first.sd)}</span>. After sweep{' '}
        <span className="num">{result.sweeps}</span>, the last one, it had mean{' '}
        <span className="num">{sweeps.last === null ? '—' : fixed(sweeps.last.mean)}</span> and sd{' '}
        <span className="num">{sweeps.last === null ? '—' : fixed(sweeps.last.sd)}</span>. The exact mean is{' '}
        <span className="num">{fixed(result.exact.mean)}</span>.
      </p>

      <p>
        EP needs a rule for when to stop. After each sweep the engine measures the largest change of any belief, in
        units of its sd. It stops when that is below a tolerance, here {sci(TOLERANCE)}, or after a fixed number of
        sweeps, the most it may make, here {MAX_SWEEPS}.
      </p>

      <Figure caption="The largest change of the belief over each sweep, on a logarithmic axis. The dashed line is the tolerance.">
        <ChangeFigure result={result} />
      </Figure>

      {result.settled ? (
        <p>
          This run stopped after <span className="num">{result.sweeps}</span> sweeps, when the largest change was{' '}
          <span className="num">{sci(sweeps.lastChange)}</span>, below the tolerance. Nothing more would change: the
          stand-ins are a fixed point of the updates, each one the answer to the others.
        </p>
      ) : (
        <p>
          This run did not settle. It made <span className="num">{result.sweeps}</span> sweeps, the most it was allowed,
          and the largest change over the last one was <span className="num">{sci(sweeps.lastChange)}</span>, above the
          tolerance of {sci(TOLERANCE)}. EP is not guaranteed to converge. Part 7 says what the library can do about it.
          Press <em>Run again</em> for other measurements; most draws settle.
        </p>
      )}

      <h2>6. The same run from start to finish</h2>

      <p>
        Now the run from the beginning, with the same measurements. At the start every stand-in is flat, so the
        approximation says nothing. The engine updates the prior first. Nothing has been heard yet, so the cavity is
        flat and the tilted distribution is the prior itself, which is a bell. The matched bell is the prior and so is
        the new stand-in. The first row of the table shows it: a bell with precision{' '}
        <span className="num">{fixed(firstUpdates[0]?.standIn.precision ?? Number.NaN, 2)}</span>, which is sd{' '}
        {PRIOR_SD}, and mean{' '}
        <span className="num">
          {firstUpdates[0] === undefined
            ? ''
            : fixed(firstUpdates[0].standIn.shift / firstUpdates[0].standIn.precision)}
        </span>
        .
      </p>

      <Figure caption="The first sweep, update by update. Cavity: what the other factors say. New stand-in: mean, with the precision in brackets. After: the approximation once the stand-in is in place.">
        <Table
          head={['update', 'factor', 'cavity', 'new stand-in', 'approximation after']}
          rows={firstUpdates.map((s, i) => [
            i + 1,
            s.factor === 0 ? 'prior' : `measurement ${s.factor} (y = ${fixed(result.points[s.factor - 1] as number)})`,
            s.cavity === null ? 'flat' : bellText(s.cavity),
            messageText(s.standIn),
            bellText(s.belief),
          ])}
        />
      </Figure>

      {second === undefined ? null : (
        <p>
          The second update is the first measurement, <span className="num">{fixed(result.points[0] as number)}</span>.
          All the other stand-ins are still flat, so its cavity is the approximation so far, the prior: mean{' '}
          <span className="num">{second.cavity === null ? '—' : fixed(second.cavity.mean)}</span>, sd{' '}
          <span className="num">{second.cavity === null ? '—' : fixed(second.cavity.sd)}</span>. Steps 2 and 3 give a
          matched bell and step 4 turns it into a stand-in with mean{' '}
          <span className="num">
            {second.standIn.precision > 0 ? fixed(second.standIn.shift / second.standIn.precision) : '—'}
          </span>{' '}
          and precision <span className="num">{fixed(second.standIn.precision)}</span>. The approximation after it has
          mean <span className="num">{second.belief === null ? '—' : fixed(second.belief.mean)}</span> and sd{' '}
          <span className="num">{second.belief === null ? '—' : fixed(second.belief.sd)}</span>.
          {lastUpdate === undefined ? null : (
            <>
              {' '}
              Each later update starts from what the one before left. After the last update of the sweep, the one for
              measurement {lastUpdate.factor}, the approximation has mean{' '}
              <span className="num">{lastUpdate.belief === null ? '—' : fixed(lastUpdate.belief.mean)}</span> and sd{' '}
              <span className="num">{lastUpdate.belief === null ? '—' : fixed(lastUpdate.belief.sd)}</span>. That is the
              end of sweep 1.
            </>
          )}
        </p>
      )}

      <p>
        The second sweep is different, because now each factor has a stand-in of its own to take out in step 1. The
        table below follows every stand-in over the sweeps. Each cell is the stand-in&rsquo;s mean and, in brackets, its
        precision. The last row is the approximation, the product of the column.
      </p>

      <Figure caption="The stand-in of every factor after some of the sweeps: mean, with the precision in brackets. A negative precision means the stand-in is not a bell. The last row is the approximation after that sweep.">
        <Table
          head={['factor', ...sweepTable.map((s) => `after sweep ${s.sweep}`)]}
          rows={[
            ...Array.from({ length: result.rows + 1 }, (_, j) => [
              j === 0 ? 'prior' : `measurement ${j} (y = ${fixed(result.points[j - 1] as number)})`,
              ...sweepTable.map((s) => messageText(s.messages[j] ?? { shift: 0, precision: 0 })),
            ]),
            ['approximation', ...sweepTable.map((s) => beliefAfter(s.sweep))],
          ]}
        />
      </Figure>

      <p>
        {result.settled
          ? 'The stand-ins change a lot in the first sweeps and by the last sweep they no longer change.'
          : `In this run the stand-ins had still not stopped changing by sweep ${result.sweeps}.`}
        {sweeps.notBells === 0 ? (
          <> At the end every measurement&rsquo;s stand-in is a proper bell, with positive precision.</>
        ) : (
          <>
            {' '}
            At the end <span className="num">{sweeps.notBells}</span> of the {result.rows} measurement stand-ins{' '}
            {sweeps.notBells === 1 ? 'has' : 'have'} a precision that is not positive.{' '}
            {sweeps.notBells === 1 ? 'It is' : 'They are'} not bells, and that is fine because their sum with the others
            is.
          </>
        )}{' '}
        What EP reports is the last row: a bell with mean <span className="num">{fixed(result.ep.mean)}</span> and sd{' '}
        <span className="num">{fixed(result.ep.sd)}</span>.
      </p>

      <p>
        Now the exact answer. The model has one unknown, so the library can compute the posterior on a fine grid of
        values of the location: the product of the prior and the {result.rows} brackets, divided by its total. Here it
        is, with EP&rsquo;s bell on top, the measurements as dots, and the location the data were made around.
      </p>

      <Figure
        wide
        caption="The exact posterior of the location (orange) and EP's bell (dashed). The dots are the measurements: blue dots were readings of the location, green dots were clutter, which EP was not told. The dotted line is the location the generator used."
      >
        <AnswerFigure result={result} />
      </Figure>

      {exact.close ? (
        <p>
          In this run EP&rsquo;s bell has mean <span className="num">{fixed(result.ep.mean)}</span> and sd{' '}
          <span className="num">{fixed(result.ep.sd)}</span>. The exact posterior has mean{' '}
          <span className="num">{fixed(result.exact.mean)}</span> and sd{' '}
          <span className="num">{fixed(result.exact.sd)}</span>. The two agree closely: the means are{' '}
          {fixed(exact.meanGap, 2)} exact sd apart, and the spread of EP&rsquo;s bell is {fixed(exact.spreadRatio, 2)}{' '}
          times the exact one. Here a single bell is a good description of the posterior.
        </p>
      ) : (
        <p>
          In this run EP&rsquo;s bell has mean <span className="num">{fixed(result.ep.mean)}</span> and sd{' '}
          <span className="num">{fixed(result.ep.sd)}</span>. The exact posterior has mean{' '}
          <span className="num">{fixed(result.exact.mean)}</span> and sd{' '}
          <span className="num">{fixed(result.exact.sd)}</span>. They differ: the means are {fixed(exact.meanGap, 2)}{' '}
          exact sd apart, and the spread of EP&rsquo;s bell is {fixed(exact.spreadRatio, 2)} times the exact one.{' '}
          {exact.humps > 1
            ? `The exact posterior has ${exact.humps} humps, because the data can be explained in more than one way, and one bell cannot cover them.`
            : 'The exact posterior is not a bell, and one bell cannot match it everywhere.'}{' '}
          {result.settled ? '' : 'This run also did not settle, so the bell shown is where it stopped.'} This is where
          EP is an approximation and not the answer.
        </p>
      )}

      {result.logEvidence === undefined ? null : (
        <p>
          EP also gives an estimate of the evidence, the total that the posterior is divided by: how probable the model
          thought the measurements were. It is built from the sizes of the tilted curves met along the way, and it comes
          with no extra computation. The run&rsquo;s estimate of the log evidence is{' '}
          <span className="num">{fixed(result.logEvidence, 2)}</span>. The grid gives{' '}
          <span className="num">{fixed(result.exact.logEvidence, 2)}</span>, a difference of{' '}
          <span className="num">{fixed(Math.abs(result.logEvidence - result.exact.logEvidence), 2)}</span>.
        </p>
      )}

      <h2>7. What EP gives, and where it goes wrong</h2>

      <p>
        EP gives a bell for each unknown, which is a mean and a spread, and an estimate of the model evidence. It does
        so in a number of updates that grows with the number of factors. It also has known weaknesses, and the library
        has a setting for each.
      </p>

      <ul>
        <li>
          <strong>Spreads that are too narrow.</strong> When unknowns are linked in loops, each factor sees a cavity
          that treats them as independent, and the spreads come out too small. In the library, the compiler packs
          unknowns that one observation reads together into one joint Gaussian vector (<code>packScalars</code>), which
          keeps their correlation. In its test cases the same models as separate scalars gave spreads two to three times
          too small. In this run, with one unknown, the spread of EP&rsquo;s bell was{' '}
          <span className="num">{fixed(exact.spreadRatio, 2)}</span> times the exact one.
        </li>
        <li>
          <strong>No convergence, and improper messages.</strong> The sweeps can swing instead of settling, and an
          update can produce an approximation with a negative variance, which is not a bell. The engine&rsquo;s{' '}
          <code>damping</code> keeps a share of the old message in each update. Its <code>improper</code> policy says
          what to do with an update that has no bell (<code>skip</code>, <code>damp</code>, <code>reset</code> or{' '}
          <code>error</code>). <code>restarts</code> runs again from the start with stronger damping when a run stalls,
          and <code>warmup</code> damps the first three sweeps, which helps models with hidden labels.
        </li>
        <li>
          <strong>One bell for a posterior with two humps.</strong> EP commits to one bell. If the exact posterior has
          two humps, EP covers one of them or both badly, and which one can depend on the start. In this run the exact
          posterior had <span className="num">{exact.humps}</span> {exact.humps === 1 ? 'hump' : 'humps'}. The
          library&rsquo;s <code>epMultiStartMethod</code> runs EP from several starting points and weights the answers
          it finds by their evidence.
        </li>
        <li>
          <strong>Fractional EP.</strong> The <code>power</code> setting uses each factor raised to a power between 0
          and 1 in steps 1 and 2. A power of 1 is the EP described here. Smaller powers make each update less eager to
          follow a single factor.
        </li>
      </ul>

      <p>
        The design of EP in omai, what each layer promises and the settings above, is written up in{' '}
        <code>docs/probprog/ep.md</code> in the repository.
      </p>

      <h2>8. Where to go next</h2>

      <p>
        The next stories run the same engine on larger models.{' '}
        <Link to={storyHref('two-humps')} className={linkClass}>
          Two humps
        </Link>{' '}
        fits a mixture of two bell curves to the waits between Old Faithful eruptions, with a hidden label for each
        eruption.{' '}
        <Link to={storyHref('when-did-the-rate-change')} className={linkClass}>
          When did the rate change?
        </Link>{' '}
        passes messages along a chain of years to find when a rate changed in a century of mining accidents. And{' '}
        <Link to={storyHref('is-this-coin-fair')} className={linkClass}>
          Is this coin fair?
        </Link>{' '}
        shows beliefs updating one flip at a time, in a model where the exact answer is a formula.
      </p>
    </>
  )
}
