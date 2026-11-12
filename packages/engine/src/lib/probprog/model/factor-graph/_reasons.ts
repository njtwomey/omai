/**
 * The catalogue of the reasons the compiler gives for refusing a node: one function per reason, taking the facts the sentence names (the
 * node, the family, a value, a parameter) and returning the `Refusal` with its code, the sentence and the facts. A mapping throws
 * `reasons.<name>(...)`, so the words live here once and a wording change is made in one place. The sentences are de facto API (the tests
 * and the people who read an issue match them), so a function does not change its text unless the text is wrong. The codes group the
 * reasons by what the model has to change (`RefusalCode` in `_refusal.ts`); the sections below follow them.
 */

import { Refusal } from './_refusal.ts'

/** What a reason names: a node, a family, a count, a list of dimensions. */
type Fact = string | number | readonly string[]

// ----- no-rule: An operation on random values that has no message rule. -----
export const comparisonOfRandoms = (op: string): Refusal =>
  new Refusal(
    'no-rule',
    `'${op}' of random values has no message rule (only a discrete variable against a known value is mapped, as a gate)`,
    { op },
  )
export const divisionByRandom = (): Refusal =>
  new Refusal('no-rule', 'division by a random value has no Gaussian message', {})
export const extremumOfItself = (which: string): Refusal =>
  new Refusal('no-rule', `the ${which} of a random value with itself`, { which })
export const indexNotDataOrCategorical = (): Refusal =>
  new Refusal(
    'no-rule',
    'at: the index must be data or a categorical latent (a gate); a computed random index has no message rule',
    {},
  )
export const inverseGammaArithmetic = (variable: string): Refusal =>
  new Refusal(
    'no-rule',
    `arithmetic on the inverse-gamma variable behind '${variable}': only a known multiple of it is mapped, and only as the squared scale of a Gaussian (the precision idiom)`,
    { variable },
  )
export const lookupNotGaussian = (variable: string): Refusal =>
  new Refusal(
    'no-rule',
    `the value looked up by the discrete variable '${variable}' is not a Gaussian one: it is mapped as the mean of a Gaussian observation or prior only`,
    { variable },
  )
export const nonGaussianArithmetic = (family: string, variable: string): Refusal =>
  new Refusal('no-rule', `arithmetic on the ${family} variable '${variable}': only Gaussian values add up`, {
    family,
    variable,
  })
export const opOfDiscrete = (op: string): Refusal =>
  new Refusal('no-rule', `'${op}' of a discrete variable has no message rule`, { op })
export const opOfDiscreteAndRandom = (op: string): Refusal =>
  new Refusal('no-rule', `'${op}' of a discrete variable and a random value has no message rule`, { op })
export const opOfGammaAndRandom = (op: string): Refusal =>
  new Refusal(
    'no-rule',
    `'${op}' of a gamma variable and a random value has no message rule: a precision is a known multiple of one gamma variable plus a known offset`,
    { op },
  )
export const opOfInverseGamma = (op: string): Refusal =>
  new Refusal('no-rule', `'${op}' of an inverse-gamma variance has no message rule`, { op })
export const opOfInverseGammaAndRandom = (op: string): Refusal =>
  new Refusal(
    'no-rule',
    `'${op}' of an inverse-gamma variance and a random value (or a non-positive one) has no message rule: only a positive known multiple of it, as the squared scale of a Gaussian`,
    { op },
  )
export const opOfPartlyRandomLookup = (op: string): Refusal =>
  new Refusal(
    'no-rule',
    `'${op}' of a value that is random in some states of a discrete variable has no message rule`,
    { op },
  )
export const opOfTwoDiscretes = (op: string, first: string, second: string): Refusal =>
  new Refusal('no-rule', `'${op}' of two different discrete variables ('${first}', '${second}') has no message rule`, {
    op,
    first,
    second,
  })
export const operatorOnRandom = (op: string): Refusal =>
  new Refusal(
    'no-rule',
    `operator '${op}' applied to a random value has no message rule (message passing handles affine maps of Gaussians, exp, square, tanh, softplus, abs, sigmoid, the rectifier and a clip by known bounds of a random value, and a link such as probit only as the likelihood of a Bernoulli observation)`,
    { op },
  )
export const randomTimesItself = (left: string): Refusal =>
  new Refusal(
    'no-rule',
    `a random value times itself (a square): the product factor takes two independent variables, and '${left}' is both`,
    { left },
  )
export const rateProductOfRandoms = (node: string): Refusal =>
  new Refusal('no-rule', `the rate of '${node}' is a product of two random values`, { node })
export const scaledPrecisionArithmetic = (variable: string): Refusal =>
  new Refusal(
    'no-rule',
    `arithmetic on the scaled precision of '${variable}': only a known multiple plus a known offset of a gamma variable is mapped, and only as the precision of a Gaussian`,
    { variable },
  )
export const squareOfGammaPrecision = (variable: string): Refusal =>
  new Refusal('no-rule', `the square of the gamma variable '${variable}' as a precision has no message rule`, {
    variable,
  })
export const stickBreakingArithmetic = (): Refusal =>
  new Refusal('no-rule', 'arithmetic on stick-breaking weights: they are only the probabilities of a categorical', {})
export const unknownOperator = (op: string): Refusal => new Refusal('no-rule', `unknown operator '${op}'`, { op })
export const whereConditionNotKnown = (): Refusal =>
  new Refusal(
    'no-rule',
    'where: the condition must be known or a Bernoulli latent; a comparison of random values has no message rule',
    {},
  )

// ----- random-parameter: A parameter that must be known but depends on a random value. -----
export const concentrationRandom = (node: string): Refusal =>
  new Refusal('random-parameter', `the concentration of '${node}' depends on a random value: it must be known`, {
    node,
  })
export const correlationConcentrationNotKnown = (node: string): Refusal =>
  new Refusal('random-parameter', `the concentration of '${node}' must be a known positive value`, { node })
export const covarianceRandomWithRandomMean = (node: string): Refusal =>
  new Refusal('random-parameter', `the covariance of '${node}' must be a known matrix when its mean is random`, {
    node,
  })
export const cutPointsShape = (node: string, dims: Fact): Refusal =>
  new Refusal(
    'shape',
    `the cut points of '${node}' must be one vector over one dimension of their own (and the observation's dimensions), they have [${dims}]`,
    { node, dims },
  )
export const exposureRandom = (node: string): Refusal =>
  new Refusal(
    'random-parameter',
    `the exposure of '${node}' is random: the poisson likelihood has a message rule for a known exposure, or a known value chosen by a discrete variable (a gate)`,
    { node },
  )
export const jointVectorInclusionRandom = (node: string): Refusal =>
  new Refusal(
    'random-parameter',
    `'${node}' is read as a vector by a matmul and has a random inclusion or scale: a joint Gaussian vector has one covariance, so they must be known`,
    { node },
  )
export const jointVectorScaleRandom = (node: string): Refusal =>
  new Refusal(
    'random-parameter',
    `'${node}' is read as a vector by a matmul and has a random scale: a joint Gaussian vector has one covariance, so its scale must be known`,
    { node },
  )
export const knownParameterRandom = (what: string, node: string, fam: string): Refusal =>
  new Refusal(
    'random-parameter',
    `the ${what} of '${node}' is random: the ${fam} likelihood has a message rule for a known ${what} only`,
    { what, node, fam },
  )
export const knownParameterRandomBecause = (what: string, node: string, message: string, fam: string): Refusal =>
  new Refusal(
    'random-parameter',
    `the ${what} of '${node}' is random (${message}): the ${fam} likelihood has a message rule for a known ${what} only`,
    { what, node, message, fam },
  )
export const logitsRandom = (node: string): Refusal =>
  new Refusal(
    'random-parameter',
    `the logits of '${node}' depend on a random value: only probs from a Dirichlet latent map`,
    { node },
  )
export const matrixRandom = (what: string, node: string): Refusal =>
  new Refusal('random-parameter', `the ${what} of '${node}' depends on a random value: a known matrix is mapped`, {
    what,
    node,
  })
export const parameterRandomForLikelihood = (p: Fact, node: string, fam: string): Refusal =>
  new Refusal(
    'random-parameter',
    `the ${p} of '${node}' is random: the ${fam} likelihood has a message rule for a known ${p} only`,
    { p, node, fam },
  )
export const priorMeanOfVectorRandom = (node: string): Refusal =>
  new Refusal('random-parameter', `'${node}' is read as a vector by a matmul, so its prior mean must be known`, {
    node,
  })
export const priorParameterRandom = (name: string, node: string, what: string): Refusal =>
  new Refusal(
    'random-parameter',
    `the ${name} of '${node}' depends on a random value: ${what} must be known for a message-passing prior`,
    { name, node, what },
  )
export const quantileLevelRandom = (node: string): Refusal =>
  new Refusal('random-parameter', `the quantile level of '${node}' is random: it must be known`, { node })
export const scaleNotKnown = (node: string): Refusal =>
  new Refusal('random-parameter', `the scale of '${node}' is not known`, { node })
export const scaleRandomInSomeStates = (node: string): Refusal =>
  new Refusal(
    'random-parameter',
    `the scale of '${node}' is random in some states of a discrete variable: it must be known`,
    { node },
  )
export const widthRandom = (node: string): Refusal =>
  new Refusal(
    'random-parameter',
    `the width of '${node}' is random: the outlier mixture has a message rule for a known width`,
    { node },
  )
export const wishartParametersRandom = (node: string): Refusal =>
  new Refusal('random-parameter', `the degrees of freedom and the scale of '${node}' must be known values`, { node })

// ----- wrong-form: A parameter or value that is not in the form the mapping takes (an idiom, a latent of a family, a linear map). -----
export const breaksNotBeta = (): Refusal =>
  new Refusal('wrong-form', 'the breaks of a stick-breaking process must be beta latents', {})
export const concentrationNotMultipleOfLatent = (node: string): Refusal =>
  new Refusal(
    'wrong-form',
    `a concentration of '${node}' is not a known multiple of a gamma or log-normal latent plus a known value`,
    { node },
  )
export const concentrationNotMultiplesOfOne = (node: string): Refusal =>
  new Refusal(
    'wrong-form',
    `the concentration of '${node}' must be known multiples of one random positive value (a gamma or log-normal latent) plus known offsets`,
    { node },
  )
export const concentrationNotRandomTimesKnown = (node: string): Refusal =>
  new Refusal(
    'wrong-form',
    `the concentration of '${node}' is not a random value times a known vector plus a known vector`,
    { node },
  )
export const dispersionNotPositiveLatent = (node: string): Refusal =>
  new Refusal(
    'wrong-form',
    `the dispersion of '${node}' is random but not a positive latent (a half-normal, half-Cauchy, log-normal or uniform prior carried on its log): the negative-binomial-dispersion factor needs the log of it`,
    { node },
  )
export const gammaRateNotMultiple = (node: string): Refusal =>
  new Refusal(
    'wrong-form',
    `the rate of '${node}' must be a known multiple of a gamma latent (the message rule integrates the gamma's rate); it is not`,
    { node },
  )
export const gateBranchNotOneLikelihood = (node: string, branch: Fact, family: string, what: string): Refusal =>
  new Refusal(
    'wrong-form',
    `branch ${branch} (${family}) of the gate '${node}' does not map to one likelihood factor on its own: ${what}`,
    { node, branch, family, what },
  )
export const gateBranchRefused = (node: string, branch: Fact, family: string, reason: string): Refusal =>
  new Refusal('wrong-form', `branch ${branch} (${family}) of the gate '${node}' is refused: ${reason}`, {
    node,
    branch,
    family,
    reason,
  })
export const gateIndexNotDiscrete = (node: string): Refusal =>
  new Refusal(
    'wrong-form',
    `the index of the gate '${node}' must be known or a discrete or Bernoulli latent: a computed random index has no message rule`,
    { node },
  )
export const gateKnownComponent = (): Refusal =>
  new Refusal(
    'wrong-form',
    'a gate selects among random values: a known component has no variable (give it a prior)',
    {},
  )
export const gateMixedFamilies = (kind: string): Refusal =>
  new Refusal('wrong-form', `a gate selects among values of one family: they are not all ${kind}`, { kind })
export const gateStateNotKnownOrGaussian = (): Refusal =>
  new Refusal('wrong-form', 'a value chosen by a discrete variable must be known or a Gaussian form in every state', {})
export const hurdleRateNotExposureExp = (node: string): Refusal =>
  new Refusal('wrong-form', `the rate of '${node}' must be \`exposure * exp(affine)\` for the hurdle Poisson`, { node })
export const keyNotPositiveLatent = (key: string, node: string): Refusal =>
  new Refusal(
    'wrong-form',
    `the ${key} of '${node}' is random but not a gamma or exponential latent, a positive latent carried on its log (a half-normal, half-Cauchy, log-normal or uniform prior) or exp of an affine value: the likelihood-pair factor needs one of those`,
    { key, node },
  )
export const locationScaleDifferentSwitches = (node: string): Refusal =>
  new Refusal('wrong-form', `the location and the scale of '${node}' are chosen by different discrete variables`, {
    node,
  })
export const lookupsDifferentStates = (node: string): Refusal =>
  new Refusal('wrong-form', `the lookups of '${node}' have different numbers of states`, { node })
export const meanIndependentOfVector = (node: string): Refusal =>
  new Refusal('wrong-form', `the mean of '${node}' does not depend on a Gaussian vector`, { node })
export const meanNotGaussianOrLookup = (node: string): Refusal =>
  new Refusal('wrong-form', `the mean of '${node}' is not a Gaussian form or a lookup`, { node })
export const meanNotKnownOrWholeVector = (node: string): Refusal =>
  new Refusal('wrong-form', `the mean of '${node}' must be a known vector or one whole Gaussian vector variable`, {
    node,
  })
export const meanNotLinearMapOfOneVector = (node: string): Refusal =>
  new Refusal('wrong-form', `the mean of '${node}' must be a linear map of one Gaussian vector`, { node })
export const meanNotLinearMapOfVector = (node: string): Refusal =>
  new Refusal('wrong-form', `the mean of '${node}' must be a linear map of another Gaussian vector`, { node })
export const meanNotOneGaussianVector = (node: string, d: Fact): Refusal =>
  new Refusal('wrong-form', `the mean of '${node}' must be one Gaussian vector of ${d} entries (or known)`, { node, d })
export const meanSpreadDifferentSwitches = (node: string): Refusal =>
  new Refusal('wrong-form', `the mean and the spread of '${node}' are chosen by different discrete variables`, { node })
export const parameterNotLatentOf = (param: string, node: string, want: Fact, fam: string): Refusal =>
  new Refusal(
    'wrong-form',
    `the ${param} of '${node}' must be a ${want} latent for the conjugate '${fam}' operator, it is not`,
    { param, node, want, fam },
  )
export const parametersNotInMessageForm = (node: string, fam: string, what: string): Refusal =>
  new Refusal('wrong-form', `the parameters of '${node}' (${fam}) are not in a form with a message rule${what}`, {
    node,
    fam,
    what,
  })
export const precisionNotGammaLatent = (node: string, what: string, idiom: string): Refusal =>
  new Refusal(
    'wrong-form',
    `the precision of '${node}' must be a gamma or exponential latent, not a ${what}: ${idiom}`,
    { node, what, idiom },
  )
export const precisionNotKnownMultiple = (node: string, message: string, idiom: string): Refusal =>
  new Refusal(
    'wrong-form',
    `the precision of '${node}' must be a known multiple of a gamma or exponential latent plus a known value (${message}); ${idiom}`,
    { node, message, idiom },
  )
export const probabilitiesNotConjugate = (node: string): Refusal =>
  new Refusal(
    'wrong-form',
    `the probabilities of '${node}' must be known or a Dirichlet latent for the conjugate 'discrete' operator`,
    { node },
  )
export const probabilityNotConjugate = (node: string): Refusal =>
  new Refusal(
    'wrong-form',
    `the probability of '${node}' must be known or a beta latent for the conjugate 'bernoulli' operator`,
    { node },
  )
export const probabilityNotLinkOrBeta = (node: string): Refusal =>
  new Refusal(
    'wrong-form',
    `the probability of '${node}' is not a link of an affine value (probit, sigmoid, a comparison) or a beta latent: no message rule`,
    { node },
  )
export const probsNotBeta = (node: string, fam: string): Refusal =>
  new Refusal(
    'wrong-form',
    `the probs of '${node}' must be a beta latent for the conjugate '${fam}' operator, it is not`,
    { node, fam },
  )
export const rateNotConjugate = (node: string, fam: string): Refusal =>
  new Refusal(
    'wrong-form',
    `the rate of '${node}' must be a gamma latent or a known multiple of one for the conjugate '${fam}' operator, it is not`,
    { node, fam },
  )
export const rhoNotKnownOrCorrelation = (node: string): Refusal =>
  new Refusal('wrong-form', `the rho of '${node}' must be known or a latent with the correlation prior`, { node })
export const rowNotChosenByDiscrete = (node: string): Refusal =>
  new Refusal('wrong-form', `the row of '${node}' must be chosen by a discrete latent, not a computed value`, { node })
export const rowsNotDirichlet = (node: string): Refusal =>
  new Refusal(
    'wrong-form',
    `the rows of the matrix behind '${node}' must be one Dirichlet latent each (a latent over the dimension of the states)`,
    { node },
  )
export const scaleNotBareGamma = (param: string, node: string): Refusal =>
  new Refusal(
    'wrong-form',
    `the ${param} of '${node}' must be 1 / sqrt(tau) with tau a gamma latent (a known multiple or offset of it is not mapped)`,
    { param, node },
  )
export const scaleNotPrecisionIdiom = (param: string, node: string): Refusal =>
  new Refusal(
    'wrong-form',
    `the ${param} of '${node}' depends on a random value that is not the precision idiom: write it as 1 / sqrt(tau) with tau a gamma latent`,
    { param, node },
  )
export const scaleNotPrecisionIdiomForOutliers = (node: string): Refusal =>
  new Refusal(
    'wrong-form',
    `the scale of '${node}' must be the precision idiom 1 / sqrt(tau) of a gamma or exponential latent for the outlier mixture`,
    { node },
  )
export const scaleNotPrecisionIdiomWithIdiom = (param: string, node: string, idiom: string): Refusal =>
  new Refusal(
    'wrong-form',
    `the ${param} of '${node}' depends on a random value that is not the precision idiom: ${idiom}`,
    { param, node, idiom },
  )
export const shareNotKnownOrBeta = (node: string): Refusal =>
  new Refusal(
    'wrong-form',
    `the share of '${node}' must be known or a beta latent (the conjugate 'bernoulli' operator on the switch of a mixture)`,
    { node },
  )
export const sharedPrecisionNotOneGamma = (node: string): Refusal =>
  new Refusal(
    'wrong-form',
    `the precisions of '${node}' must be multiples of one gamma variable plus constants (a vector with several random precisions is the ARD form, one variable per coefficient)`,
    { node },
  )
export const sharedPrecisionWithMean = (node: string): Refusal =>
  new Refusal(
    'wrong-form',
    `'${node}' has a shared random precision and a mean: the vector with a smoothing parameter must have a zero known mean`,
    { node },
  )
export const weibullScaleNotLog = (node: string): Refusal =>
  new Refusal(
    'wrong-form',
    `the scale of '${node}' is not exp of an affine value: the Weibull with a random shape needs the log link`,
    { node },
  )
export const weightNotBeta = (node: string): Refusal =>
  new Refusal('wrong-form', `the weight of '${node}' must be a beta latent for the 'mixture-weight' operator`, { node })
export const zeroProbabilityNotKnownOrBeta = (node: string): Refusal =>
  new Refusal('wrong-form', `the zero probability of '${node}' must be known or a beta latent`, { node })
export const zeroTruncatedRateForm = (node: string): Refusal =>
  new Refusal('wrong-form', `the rate of '${node}' must be \`exposure * exp(affine)\` for the zero-truncated Poisson`, {
    node,
  })

// ----- registry: The operator registry lacks an operator the compiler needs, or the engine refuses a factor the compiler built. -----
export const noOperator = (name: string): Refusal =>
  new Refusal('registry', `the message-passing registry has no factor operator '${name}'`, { name })

/** The engine refuses a factor the compiler built: the message is what `compileGraph` says of it (a parameter, an arity, a family a port does not take, the operator's own check). */
export const factorRejected = (factor: string, message: string): Refusal =>
  new Refusal('registry', `the factor '${factor}' is not one the engine takes: ${message}`, { factor, message })

// ----- random-value: The evaluation of a known expression reached a random value. -----
export const observedValueRandom = (node: string, message: string): Refusal =>
  new Refusal('random-value', `the value observed by '${node}' is random: ${message}`, { node, message })
export const randomValueWhereKnown = (): Refusal =>
  new Refusal('random-value', 'a random value where a known one is needed', {})
export const randomWhereKnown = (name: string): Refusal =>
  new Refusal('random-value', `'${name}' is random where a known value is needed`, { name })

// ----- node: A node that has no factor (a param, a log-density term) or that depends on one that failed. -----
export const dependsOnFailed = (name: string): Refusal =>
  new Refusal('node', `depends on '${name}', which could not be compiled`, { name })
export const logDensityTermHasNoOperator = (node: string): Refusal =>
  new Refusal(
    'node',
    `'${node}' is an arbitrary log-density term: no message operator exists for it (a numeric factor operator could be given)`,
    { node },
  )
export const paramNodeHasNoFactor = (node: string): Refusal =>
  new Refusal('node', `'${node}' is a param (a learnable value, not a random variable): it has no factor`, { node })

// ----- missing: A parameter the family needs is not given. -----
export const concentrationMissing = (node: string): Refusal =>
  new Refusal('missing', `'${node}': the concentration is missing`, { node })
export const meanMissing = (node: string): Refusal => new Refusal('missing', `'${node}': the mean is missing`, { node })
export const parameterMissing = (node: string, param: string): Refusal =>
  new Refusal('missing', `'${node}': ${param} is missing`, { node, param })
export const parameterMissingFor = (node: string, param: string): Refusal =>
  new Refusal('missing', `'${node}': the ${param} is missing`, { node, param })
export const parameterMissingIn = (family: string, name: string): Refusal =>
  new Refusal('missing', `${family}: the parameter '${name}' is missing`, { family, name })
export const probabilityMissing = (node: string): Refusal =>
  new Refusal('missing', `'${node}': the probability is missing`, { node })
export const probsOrLogitsMissing = (node: string): Refusal =>
  new Refusal('missing', `'${node}': give probs or logits`, { node })
export const rhoMissing = (node: string): Refusal => new Refusal('missing', `'${node}': rho is missing`, { node })

// ----- shape: Dimensions or sizes of a value that do not fit. -----
export const classDimensionMissing = (param: string, node: string, dims: Fact): Refusal =>
  new Refusal(
    'shape',
    `the ${param} of '${node}' must carry the class dimension declared by \`over\` (categorical(..., { over })), it has [${dims}]`,
    { param, node, dims },
  )
export const covarianceNotSquare = (node: string, n: Fact): Refusal =>
  new Refusal('shape', `the covariance of '${node}' is not ${n} by ${n}`, { node, n })
export const customReturnedWrongCount = (name: string, returned: Fact, expected: Fact): Refusal =>
  new Refusal('shape', `the custom function '${name}' returned ${returned} values, not ${expected}`, {
    name,
    returned,
    expected,
  })
export const matmulShapes = (first: Fact, second: Fact): Refusal =>
  new Refusal('shape', `matmul: need [..., d] times [d] or [d, k], got [${first}] and [${second}]`, { first, second })
export const matrixEntriesVersusVector = (op: string, entries: Fact, n: Fact): Refusal =>
  new Refusal('shape', `${op}: the matrix has ${entries} entries for a vector of ${n}`, { op, entries, n })
export const meanNotVectorOf = (node: string, d: Fact): Refusal =>
  new Refusal('shape', `the mean of '${node}' must be a vector of ${d} (per row, when it varies)`, { node, d })
export const observedPairsSize = (node: string, dims: Fact): Refusal =>
  new Refusal('shape', `'${node}' is observed as vectors of 2 entries over [${dims}]`, { node, dims })
export const observedVectorsSize = (node: string, d: Fact, dims: Fact): Refusal =>
  new Refusal('shape', `'${node}' is observed as vectors of ${d} entries over [${dims}]`, { node, d, dims })
export const valueDoesNotFitDims = (dim: string, dims: Fact): Refusal =>
  new Refusal('shape', `a value over '${dim}' does not fit the dimensions [${dims}]`, { dim, dims })
export const valueOverDimDoesNotFit = (dim: string, dims: Fact): Refusal =>
  new Refusal('shape', `a parameter over '${dim}' does not fit [${dims}]`, { dim, dims })
export const vectorSizeVersusMatrix = (node: string, d: Fact, wishart: string, order: Fact): Refusal =>
  new Refusal('shape', `'${node}' has ${d} entries per vector, the matrix '${wishart}' is ${order} by ${order}`, {
    node,
    d,
    wishart,
    order,
  })
export const wishartScaleNotSquare = (node: string): Refusal =>
  new Refusal('shape', `the scale of '${node}' must be a square matrix`, { node })

// ----- structure: A shape of model the compiler does not map. -----
export const atNeedsDimension = (): Refusal =>
  new Refusal('structure', 'at: the first operand needs a dimension to index', {})
export const cannotBeRecorded = (node: string, what: string): Refusal =>
  new Refusal('structure', `'${node}' is a ${what}: it cannot be recorded as a value`, { node, what })
export const dependsOnDiscrete = (name: string): Refusal =>
  new Refusal('structure', `'${name}' depends on a discrete variable: only a table over scalar nodes handles that`, {
    name,
  })
export const discreteParentsOverPlate = (node: string, dims: Fact): Refusal =>
  new Refusal(
    'structure',
    `'${node}' is a Bernoulli over [${dims}] with discrete parents: tables over plated nodes are not mapped`,
    { node, dims },
  )
export const eventAxesInScalar = (): Refusal =>
  new Refusal('structure', 'a value with event axes cannot stand in a scalar expression', {})
export const bivariateVectorNeedsAxis = (node: string): Refusal =>
  new Refusal(
    'structure',
    `'${node}' is a latent bivariateNormal without an axis for the vector: write dims: [group, coef] (or [coef] for one vector)`,
    { node },
  )
export const bivariateVectorSize = (node: string, size: number): Refusal =>
  new Refusal(
    'structure',
    `'${node}' is a bivariateNormal over an axis of ${size} entries: a bivariate vector has two`,
    {
      node,
      size,
    },
  )
export const oneCorrelationOverDims = (node: string, dims: Fact): Refusal =>
  new Refusal('structure', `'${node}' is a correlation over [${dims}]: one correlation is mapped`, { node, dims })
export const oneMatrixOverDims = (node: string, family: string, dims: Fact): Refusal =>
  new Refusal('structure', `'${node}' is a ${family} over [${dims}]: one matrix is mapped`, { node, family, dims })
export const oneVectorOverOneDimension = (node: string): Refusal =>
  new Refusal('structure', `'${node}' is a multivariateNormal: one vector over one dimension is mapped`, { node })
export const pickNeedsDimension = (): Refusal =>
  new Refusal('structure', 'pick: the values need a dimension to pick along', {})
export const precisionGivenForNormal = (node: string): Refusal =>
  new Refusal('structure', `'${node}' is a multivariateNormal with a precision: give a covariance or scaleTril`, {
    node,
  })
export const siteHasNoDiscrete = (site: string): Refusal =>
  new Refusal('structure', `'${site}' has no discrete variable`, { site })
export const stickBreakingNeedsDimension = (): Refusal =>
  new Refusal('structure', 'stickBreaking needs a vector of breaks over a named dimension', {})
export const stickBreakingNotRecorded = (node: string): Refusal =>
  new Refusal('structure', `'${node}' is a stick-breaking weight vector`, { node })
export const subsampledByBatch = (node: string): Refusal =>
  new Refusal('structure', `'${node}' is subsampled by the batch option: a factor graph is built over all the rows`, {
    node,
  })
export const wishartReadAsWrongKind = (node: string, wishart: string, key: string, kind: string): Refusal =>
  new Refusal('structure', `'${node}' reads '${wishart}' as its ${key}, but '${wishart}' is ${kind}`, {
    node,
    wishart,
    key,
    kind,
  })
export const wishartVectorNeedsAxis = (node: string): Refusal =>
  new Refusal(
    'structure',
    `'${node}' is a multivariate normal with a Wishart precision: give the vector's axis a named dimension (dims: [..., axis])`,
    { node },
  )

// ----- support: A value, count or size outside what the model or the family accepts. -----
export const choicesVersusRows = (node: string, choices: Fact, rows: Fact): Refusal =>
  new Refusal('support', `'${node}' has ${choices} choices of row and its matrix ${rows} rows`, { node, choices, rows })
export const classOutsideRange = (node: string, y: Fact, last: Fact): Refusal =>
  new Refusal('support', `'${node}': the class ${y} is outside 0 to ${last}`, { node, y, last })
export const countNotOnSupport = (node: string, fam: string, y: Fact): Refusal =>
  new Refusal('support', `'${node}' is ${fam}: the count ${y} is not on its support`, { node, fam, y })
export const exposureNegative = (node: string): Refusal =>
  new Refusal('support', `the exposure of '${node}' must be non-negative`, { node })
export const gateIndexOutsideBranches = (node: string, index: Fact, branches: Fact): Refusal =>
  new Refusal('support', `the index ${index} of the gate '${node}' is outside its ${branches} branches`, {
    node,
    index,
    branches,
  })
export const gateStatesVersusBranches = (node: string, states: Fact, branches: Fact): Refusal =>
  new Refusal('support', `the index of the gate '${node}' has ${states} states and the gate ${branches} branches`, {
    node,
    states,
    branches,
  })
export const hurdleNotCount = (node: string, y: Fact): Refusal =>
  new Refusal('support', `'${node}' is a hurdle count: ${y} is not a count`, { node, y })
export const indexOutOfRange = (index: Fact, dim: string): Refusal =>
  new Refusal('support', `at: index ${index} is out of range for '${dim}'`, { index, dim })
export const keyNotPositive = (key: string, node: string, value: Fact): Refusal =>
  new Refusal('support', `the ${key} of '${node}' is ${value}: it must be positive`, { key, node, value })
export const likelihoodRejects = (node: string, likelihood: string, why: string): Refusal =>
  new Refusal('support', `'${node}' (${likelihood}): ${why}`, { node, likelihood, why })
export const logNormalOutcome = (node: string, y: Fact): Refusal =>
  new Refusal('support', `'${node}' is log-normal: the outcome ${y} must be positive`, { node, y })
export const negativeBinomialOutcome = (node: string): Refusal =>
  new Refusal('support', `'${node}' (negative binomial): the outcome must be a non-negative integer`, { node })
export const outcomeOutsideCategories = (node: string, y: Fact, size: Fact): Refusal =>
  new Refusal('support', `'${node}' observes ${y}, outside its ${size} categories`, { node, y, size })
export const parameterNotPositive = (param: string, node: string): Refusal =>
  new Refusal('support', `${param} of '${node}' must be positive`, { param, node })
export const poissonLogRejects = (node: string, why: string): Refusal =>
  new Refusal('support', `'${node}' (poisson-log): ${why}`, { node, why })
export const precisionNotNonNegative = (node: string, scale: Fact, offset: Fact): Refusal =>
  new Refusal(
    'support',
    `the precision of '${node}' is ${scale} times a gamma variable plus ${offset}: both must be non-negative, not both zero`,
    { node, scale, offset },
  )
export const precisionNotNonNegativeMultiple = (node: string): Refusal =>
  new Refusal('support', `the precision of '${node}' must be a non-negative multiple of a gamma variable`, { node })
export const previousStatesVersusRows = (node: string, states: Fact, rows: Fact): Refusal =>
  new Refusal('support', `'${node}' has ${states} previous states and its matrix ${rows} rows`, { node, states, rows })
export const probabilityOutsideUnit = (node: string, p1: Fact): Refusal =>
  new Refusal('support', `'${node}': a probability of ${p1} is outside [0, 1]`, { node, p1 })
export const rhoOutsideUnit = (node: string): Refusal =>
  new Refusal('support', `rho of '${node}' must be in (-1, 1)`, { node })
export const scaleNotPositive = (node: string, scale: Fact): Refusal =>
  new Refusal('support', `the scale of '${node}' is ${scale}: it must be positive`, { node, scale })
export const scalesNotPositive = (node: string): Refusal =>
  new Refusal('support', `the scales of '${node}' must be positive`, { node })
export const shareOutsideUnit = (node: string, p: Fact): Refusal =>
  new Refusal('support', `the share of '${node}' is ${p}: it must be strictly between 0 and 1`, { node, p })
export const sideScaleNotPositive = (node: string, side: string, scale: Fact): Refusal =>
  new Refusal('support', `the ${node} scale of '${side}' is ${scale}: it must be positive`, { node, side, scale })
export const sigmaNotPositive = (node: string): Refusal =>
  new Refusal('support', `the sigma of '${node}' must be positive`, { node })
export const transitionMatrixSize = (node: string, entries: Fact, states: Fact): Refusal =>
  new Refusal('support', `the transition matrix of '${node}' has ${entries} entries for ${states} states`, {
    node,
    entries,
    states,
  })
export const valueNotPositive = (node: string, fam: string, y: Fact): Refusal =>
  new Refusal('support', `'${node}' is ${fam}: the value ${y} must be positive`, { node, fam, y })
export const widthNotPositive = (node: string, width: Fact): Refusal =>
  new Refusal('support', `the width of '${node}' is ${width}: it must be positive`, { node, width })
export const wishartDegreesOfFreedom = (node: string, df: Fact, d1: Fact): Refusal =>
  new Refusal('support', `'${node}' has ${df} degrees of freedom: more than ${d1} are needed`, { node, df, d1 })

// ----- unbound: Data that is not bound, a dimension without a size, missing values. -----
export const dataNotBound = (name: string): Refusal => new Refusal('unbound', `data '${name}' is not bound`, { name })
export const dimensionHasNoSize = (dim: string): Refusal =>
  new Refusal('unbound', `the dimension '${dim}' has no size (bind data for it or fix its size)`, { dim })
export const missingValuesPolicy = (node: string, count: Fact, policy: string): Refusal =>
  new Refusal(
    'unbound',
    `'${node}' has ${count} missing value(s) with the policy '${policy}': only 'ignore' is mapped (the entries drop out of the graph)`,
    { node, count, policy },
  )

// ----- constraint: A constraint beyond the family's own support. -----
export const boundNotOneNumber = (node: string): Refusal =>
  new Refusal('constraint', `the bound of the constraint of '${node}' must be one known number`, { node })
export const constraintBeyondSupport = (node: string, kind: string): Refusal =>
  new Refusal(
    'constraint',
    `'${node}' has the constraint '${kind}': a message family lives on the family's own support, so a constraint beyond it is not expressible`,
    { node, kind },
  )
export const constraintNotMapped = (node: string, kind: string): Refusal =>
  new Refusal(
    'constraint',
    `'${node}' has the constraint '${kind}': only ordered and bounds on a Gaussian are mapped`,
    { node, kind },
  )
export const orderedOverSeveralDims = (node: string, dims: Fact): Refusal =>
  new Refusal('constraint', `'${node}' is ordered over [${dims}]: one dimension is mapped`, { node, dims })

// ----- family: A distribution family with no mapping. -----
export const familyHasNoMessageFamily = (node: string, fam: string): Refusal =>
  new Refusal(
    'family',
    `'${node}' is ${fam}: the engine has message families gaussian, gaussian-mv, gamma, beta, bernoulli, discrete, dirichlet, normal-gamma, wishart and normal-wishart, and none for ${fam}`,
    { node, fam },
  )
export const observedFamilyUnmapped = (node: string, fam: string): Refusal =>
  new Refusal(
    'family',
    `'${node}' is observed as ${fam}: the likelihoods mapped are normal, logNormal, studentT, laplace, asymmetricLaplace, orderedProbit, orderedLogistic, categoricalProbit, normalOutliers, beta, bernoulli, binomial, poisson, zeroTruncatedPoisson, hurdlePoisson, normalPair, exponential, gamma, geometric, negativeBinomial, weibull, gate(...) and categorical`,
    { node, fam },
  )

// ----- opaque: A custom function the graph cannot see through. -----
export const customNotAffine = (name: string): Refusal =>
  new Refusal('opaque', `the custom function '${name}' is not affine in its random argument`, { name })
export const customNotSupplied = (name: string): Refusal =>
  new Refusal('opaque', `the custom function '${name}' was not supplied`, { name })
export const customOpaque = (name: string): Refusal =>
  new Refusal(
    'opaque',
    `the custom function '${name}' is opaque to message passing (it is read as a matrix only when it has one random vector argument, known others and a vector result)`,
    { name },
  )
