/**
 * The provenance and builders shared by the datasets vendored from the UCI Machine Learning Repository: the table of
 * their titles, descriptions, sizes, citations and source files, the metadata stamp every one carries, and the three
 * builders that turn a generated payload into a classification, regression or table dataset.
 */

import { OmaiError } from '../base/index.ts'
import {
  type ClassificationData,
  type DatasetKind,
  type DatasetMeta,
  type RegressionData,
  type TableData,
  table,
  xy,
  xyClasses,
} from '../data/index.ts'
import { fromStorage } from '../tensor/index.ts'

/** The licence of every dataset of the UCI repository vendored here. */
export const UCI_LICENCE = 'CC BY 4.0 (UCI Machine Learning Repository)'

/** What the registry and the loaders say about one UCI dataset. */
export interface UciInfo {
  readonly title: string
  readonly description: string
  readonly kind: Extract<DatasetKind, 'classification' | 'regression' | 'table'>
  /** The number of rows in the vendored payload (a subsample's, where the full file is larger). */
  readonly size: number
  /** The dataset's id on the repository. */
  readonly uciId: number
  /** The dataset's name in the repository's URLs. */
  readonly slug: string
  readonly citation: string
  /** The published file or files the payload was generated from. */
  readonly files: string
}

/** The datasets vendored from the UCI repository by loader name. */
export const UCI_INFO: Readonly<Record<string, UciInfo>> = {
  banknoteAuthentication: {
    title: 'Banknote authentication',
    description: 'Four wavelet statistics of 1372 banknote images, genuine or forged.',
    kind: 'classification',
    size: 1372,
    uciId: 267,
    slug: 'banknote+authentication',
    citation:
      'Lohweg (2012), Banknote Authentication, UCI Machine Learning Repository, https://doi.org/10.24432/C55P57.',
    files: 'data_banknote_authentication.txt',
  },
  ionosphere: {
    title: 'Ionosphere',
    description: '351 radar returns from the ionosphere as 34 numbers each, good or bad (Sigillito et al., 1989).',
    kind: 'classification',
    size: 351,
    uciId: 52,
    slug: 'ionosphere',
    citation:
      'Sigillito, Wing, Hutton and Baker (1989), Ionosphere, UCI Machine Learning Repository, https://doi.org/10.24432/C5W01B.',
    files: 'ionosphere.data',
  },
  sonar: {
    title: 'Sonar (mines versus rocks)',
    description: '208 sonar returns in 60 energy bands, from metal cylinders or rocks (Gorman and Sejnowski, 1988).',
    kind: 'classification',
    size: 208,
    uciId: 151,
    slug: 'connectionist+bench+sonar+mines+vs+rocks',
    citation:
      'Sejnowski and Gorman (1988), Connectionist Bench (Sonar, Mines vs. Rocks), UCI Machine Learning Repository, https://doi.org/10.24432/C5T01Q.',
    files: 'sonar.all-data',
  },
  glassIdentification: {
    title: 'Glass identification',
    description: 'Refractive index and oxide content of 214 glass samples of six types (German, 1987).',
    kind: 'classification',
    size: 214,
    uciId: 42,
    slug: 'glass+identification',
    citation: 'German (1987), Glass Identification, UCI Machine Learning Repository, https://doi.org/10.24432/C5WW2P.',
    files: 'glass.data',
  },
  seeds: {
    title: 'Seeds',
    description: 'Seven geometric measurements of 210 wheat kernels of three varieties (Charytanowicz et al., 2010).',
    kind: 'classification',
    size: 210,
    uciId: 236,
    slug: 'seeds',
    citation:
      'Charytanowicz, Niewczas, Kulczycki, Kowalski and Lukasik (2010), Seeds, UCI Machine Learning Repository, https://doi.org/10.24432/C5H30K.',
    files: 'seeds_dataset.txt',
  },
  ecoli: {
    title: 'E. coli protein localisation',
    description: 'Seven scores of 336 E. coli proteins in eight localisation sites (Nakai).',
    kind: 'classification',
    size: 336,
    uciId: 39,
    slug: 'ecoli',
    citation: 'Nakai (1996), Ecoli, UCI Machine Learning Repository, https://doi.org/10.24432/C5388M.',
    files: 'ecoli.data',
  },
  yeast: {
    title: 'Yeast protein localisation',
    description: 'Eight scores of 1484 yeast proteins in ten localisation sites (Nakai).',
    kind: 'classification',
    size: 1484,
    uciId: 110,
    slug: 'yeast',
    citation: 'Nakai (1991), Yeast, UCI Machine Learning Repository, https://doi.org/10.24432/C5KG68.',
    files: 'yeast.data',
  },
  haberman: {
    title: "Haberman's survival",
    description:
      'Age, year of operation and node count of 306 breast-cancer patients, alive five years later or not (Haberman, 1976).',
    kind: 'classification',
    size: 306,
    uciId: 43,
    slug: 'haberman+s+survival',
    citation: "Haberman (1976), Haberman's Survival, UCI Machine Learning Repository, https://doi.org/10.24432/C5XK51.",
    files: 'haberman.data',
  },
  balanceScale: {
    title: 'Balance scale',
    description: 'All 625 settings of weights and distances on a balance scale, and which side tips (Siegler, 1976).',
    kind: 'classification',
    size: 625,
    uciId: 12,
    slug: 'balance+scale',
    citation: 'Siegler (1976), Balance Scale, UCI Machine Learning Repository, https://doi.org/10.24432/C5488X.',
    files: 'balance-scale.data',
  },
  vehicleSilhouettes: {
    title: 'Vehicle silhouettes',
    description: '18 shape features of the silhouettes of 846 vehicles of four types (Statlog).',
    kind: 'classification',
    size: 846,
    uciId: 149,
    slug: 'statlog+vehicle+silhouettes',
    citation:
      'Mowforth and Shepherd (1987), Statlog (Vehicle Silhouettes), UCI Machine Learning Repository, https://doi.org/10.24432/C5HG6N.',
    files: 'xaa.dat to xai.dat',
  },
  wholesaleCustomers: {
    title: 'Wholesale customers',
    description:
      'Yearly spending in six product categories of 440 clients of a Portuguese wholesaler, by channel (Cardoso).',
    kind: 'classification',
    size: 440,
    uciId: 292,
    slug: 'wholesale+customers',
    citation: 'Cardoso (2013), Wholesale customers, UCI Machine Learning Repository, https://doi.org/10.24432/C5030X.',
    files: 'Wholesale customers data.csv',
  },
  lenses: {
    title: 'Contact lenses',
    description:
      'The 24 combinations of four patient attributes and whether to fit hard, soft or no contact lenses (Cendrowska, 1987).',
    kind: 'classification',
    size: 24,
    uciId: 58,
    slug: 'lenses',
    citation: 'Cendrowska (1987), Lenses, UCI Machine Learning Repository, https://doi.org/10.24432/C5K88Z.',
    files: 'lenses.data',
  },
  spambase: {
    title: 'Spambase (subsample)',
    description:
      '57 word, character and capital-letter statistics of 1000 e-mails, spam or not: a stratified subsample of the 4601 of Spambase.',
    kind: 'classification',
    size: 1000,
    uciId: 94,
    slug: 'spambase',
    citation:
      'Hopkins, Reeber, Forman and Suermondt (1999), Spambase, UCI Machine Learning Repository, https://doi.org/10.24432/C53G6X.',
    files: 'spambase.data',
  },
  concreteStrength: {
    title: 'Concrete compressive strength',
    description:
      'Compressive strength in MPa of 1030 concrete mixes from the amounts of eight ingredients and the age (Yeh, 1998).',
    kind: 'regression',
    size: 1030,
    uciId: 165,
    slug: 'concrete+compressive+strength',
    citation:
      'Yeh (1998), Concrete Compressive Strength, UCI Machine Learning Repository, https://doi.org/10.24432/C5PK67.',
    files: 'data.csv',
  },
  energyEfficiency: {
    title: 'Energy efficiency',
    description:
      'Heating and cooling loads of 768 simulated buildings from eight design attributes (Tsanas and Xifara, 2012).',
    kind: 'regression',
    size: 768,
    uciId: 242,
    slug: 'energy+efficiency',
    citation:
      'Tsanas and Xifara (2012), Energy Efficiency, UCI Machine Learning Repository, https://doi.org/10.24432/C51307.',
    files: 'data.csv',
  },
  yachtHydrodynamics: {
    title: 'Yacht hydrodynamics',
    description: 'Residuary resistance of 308 sailing-yacht hulls from six hull coefficients (Gerritsma et al., 1981).',
    kind: 'regression',
    size: 308,
    uciId: 243,
    slug: 'yacht+hydrodynamics',
    citation:
      'Gerritsma, Onnink and Versluis (1981), Yacht Hydrodynamics, UCI Machine Learning Repository, https://doi.org/10.24432/C5XG7R.',
    files: 'yacht_hydrodynamics.data',
  },
  airfoilSelfNoise: {
    title: 'Airfoil self-noise',
    description:
      'Sound pressure level of 1503 NASA wind-tunnel airfoil tests from five attributes (Brooks et al., 1989).',
    kind: 'regression',
    size: 1503,
    uciId: 291,
    slug: 'airfoil+self+noise',
    citation:
      'Brooks, Pope and Marcolini (1989), Airfoil Self-Noise, UCI Machine Learning Repository, https://doi.org/10.24432/C5VW2C.',
    files: 'airfoil_self_noise.dat',
  },
  wineQualityRed: {
    title: 'Wine quality (red)',
    description:
      'Eleven physicochemical measurements of 1599 red vinho verde wines and a median taste score from 0 to 10 (Cortez et al., 2009).',
    kind: 'regression',
    size: 1599,
    uciId: 186,
    slug: 'wine+quality',
    citation:
      'Cortez, Cerdeira, Almeida, Matos and Reis (2009), Wine Quality, UCI Machine Learning Repository, https://doi.org/10.24432/C56S3T.',
    files: 'winequality-red.csv',
  },
  wineQualityWhite: {
    title: 'Wine quality (white)',
    description:
      'Eleven physicochemical measurements of 4898 white vinho verde wines and a median taste score from 0 to 10 (Cortez et al., 2009).',
    kind: 'regression',
    size: 4898,
    uciId: 186,
    slug: 'wine+quality',
    citation:
      'Cortez, Cerdeira, Almeida, Matos and Reis (2009), Wine Quality, UCI Machine Learning Repository, https://doi.org/10.24432/C56S3T.',
    files: 'winequality-white.csv',
  },
  autoMpg: {
    title: 'Auto MPG',
    description:
      'Fuel economy of 398 cars of model years 1970 to 1982 from their design, with 6 missing horsepower values (Quinlan, 1993).',
    kind: 'table',
    size: 398,
    uciId: 9,
    slug: 'auto+mpg',
    citation: 'Quinlan (1993), Auto MPG, UCI Machine Learning Repository, https://doi.org/10.24432/C5859H.',
    files: 'auto-mpg.data',
  },
  abalone: {
    title: 'Abalone',
    description:
      'Sex and seven measurements of 4177 abalone, with the number of shell rings that gives the age (Nash et al., 1994).',
    kind: 'table',
    size: 4177,
    uciId: 1,
    slug: 'abalone',
    citation:
      'Nash, Sellers, Talbot, Cawthorn and Ford (1994), Abalone, UCI Machine Learning Repository, https://doi.org/10.24432/C55C7W.',
    files: 'abalone.data',
  },
  heartDisease: {
    title: 'Heart disease (Cleveland)',
    description:
      'Thirteen clinical attributes of 303 patients of the Cleveland Clinic and the degree of heart disease, with 6 missing values (Detrano et al., 1989).',
    kind: 'table',
    size: 303,
    uciId: 45,
    slug: 'heart+disease',
    citation:
      'Janosi, Steinbrunn, Pfisterer and Detrano (1989), Heart Disease, UCI Machine Learning Repository, https://doi.org/10.24432/C52P4X.',
    files: 'processed.cleveland.data',
  },
  hepatitis: {
    title: 'Hepatitis',
    description:
      'Nineteen clinical attributes of 155 hepatitis patients and whether they lived, with 167 missing values.',
    kind: 'table',
    size: 155,
    uciId: 46,
    slug: 'hepatitis',
    citation: 'Gong (1988), Hepatitis, UCI Machine Learning Repository, https://doi.org/10.24432/C5Q59J.',
    files: 'hepatitis.data',
  },
  zoo: {
    title: 'Zoo',
    description: 'Sixteen attributes of 101 animals in seven classes (Forsyth, 1990).',
    kind: 'table',
    size: 101,
    uciId: 111,
    slug: 'zoo',
    citation: 'Forsyth (1990), Zoo, UCI Machine Learning Repository, https://doi.org/10.24432/C5R59V.',
    files: 'zoo.data',
  },
  ticTacToe: {
    title: 'Tic-tac-toe endgame',
    description: 'All 958 final boards of tic-tac-toe where x moved first, and whether x won (Aha, 1991).',
    kind: 'table',
    size: 958,
    uciId: 101,
    slug: 'tic+tac+toe+endgame',
    citation: 'Aha (1991), Tic-Tac-Toe Endgame, UCI Machine Learning Repository, https://doi.org/10.24432/C5688J.',
    files: 'tic-tac-toe.data',
  },
  carEvaluation: {
    title: 'Car evaluation',
    description: 'All 1728 combinations of six car attributes and their acceptability (Bohanec and Rajkovic, 1988).',
    kind: 'table',
    size: 1728,
    uciId: 19,
    slug: 'car+evaluation',
    citation: 'Bohanec (1988), Car Evaluation, UCI Machine Learning Repository, https://doi.org/10.24432/C5JP48.',
    files: 'car.data',
  },
  congressionalVoting: {
    title: 'Congressional voting records',
    description: 'The 1984 votes of 435 US House members on 16 issues and their party, with 392 votes not yea or nay.',
    kind: 'table',
    size: 435,
    uciId: 105,
    slug: 'congressional+voting+records',
    citation:
      'Schlimmer (1987), Congressional Voting Records, UCI Machine Learning Repository, https://doi.org/10.24432/C5C01P.',
    files: 'house-votes-84.data',
  },
  mushroom: {
    title: 'Mushroom (subsample)',
    description:
      'Twenty-two categorical attributes of 2000 mushrooms, edible or poisonous: a stratified subsample of the 8124 of the UCI mushroom data.',
    kind: 'table',
    size: 2000,
    uciId: 73,
    slug: 'mushroom',
    citation: 'Schlimmer (1981), Mushroom, UCI Machine Learning Repository, https://doi.org/10.24432/C5959T.',
    files: 'agaricus-lepiota.data',
  },
  adultIncome: {
    title: 'Adult census income (subsample)',
    description:
      'Fourteen census attributes of 2500 US adults and whether they earn over 50K a year: a stratified subsample of the 32561 of the UCI adult training file, with missing values.',
    kind: 'table',
    size: 2500,
    uciId: 2,
    slug: 'adult',
    citation: 'Becker and Kohavi (1996), Adult, UCI Machine Learning Repository, https://doi.org/10.24432/C5XW20.',
    files: 'adult.data',
  },
  dermatology: {
    title: 'Dermatology',
    description:
      'Thirty-four clinical and histopathological attributes of 366 patients with six erythemato-squamous diseases, with 8 missing ages (Guvenir et al., 1998).',
    kind: 'table',
    size: 366,
    uciId: 33,
    slug: 'dermatology',
    citation:
      'Ilter and Guvenir (1998), Dermatology, UCI Machine Learning Repository, https://doi.org/10.24432/C5FK5P.',
    files: 'dermatology.data',
  },
  forestFires: {
    title: 'Forest fires',
    description:
      'Weather and fire-index readings for 517 forest fires in Montesinho park, Portugal, and the area burnt (Cortez and Morais, 2007).',
    kind: 'table',
    size: 517,
    uciId: 162,
    slug: 'forest+fires',
    citation:
      'Cortez and Morais (2007), Forest Fires, UCI Machine Learning Repository, https://doi.org/10.24432/C5D88D.',
    files: 'forestfires.csv',
  },
  studentMath: {
    title: 'Student performance (mathematics)',
    description:
      'Thirty-three attributes of 395 Portuguese secondary-school students and their final mathematics grade (Cortez and Silva, 2008).',
    kind: 'table',
    size: 395,
    uciId: 320,
    slug: 'student+performance',
    citation: 'Cortez (2008), Student Performance, UCI Machine Learning Repository, https://doi.org/10.24432/C5TG7T.',
    files: 'student.zip/student-mat.csv',
  },
  servo: {
    title: 'Servo',
    description: 'Rise time of 167 servomechanisms from two gain settings and two mechanical linkages (Quinlan).',
    kind: 'table',
    size: 167,
    uciId: 87,
    slug: 'servo',
    citation: 'Ulrich (1986), Servo, UCI Machine Learning Repository, https://doi.org/10.24432/C5Q30F.',
    files: 'servo.data',
  },
  bikeSharingDaily: {
    title: 'Bike sharing (daily)',
    description:
      "Weather, season and calendar for 731 days of a Washington, D.C. bike-share system, and the day's rentals (Fanaee-T and Gama, 2013).",
    kind: 'table',
    size: 731,
    uciId: 275,
    slug: 'bike+sharing+dataset',
    citation: 'Fanaee-T (2013), Bike Sharing, UCI Machine Learning Repository, https://doi.org/10.24432/C5W894.',
    files: 'day.csv',
  },
}

/** The page of a UCI dataset on the repository. */
export function uciUrl(name: string): string {
  const info = UCI_INFO[name]
  if (info === undefined) throw new OmaiError('value', `uciUrl: no UCI dataset '${name}'`)
  return `https://archive.ics.uci.edu/dataset/${info.uciId}/${info.slug}`
}

/** The source line of a UCI dataset's metadata. */
export function uciSource(name: string): string {
  const info = UCI_INFO[name]
  if (info === undefined) throw new OmaiError('value', `uciSource: no UCI dataset '${name}'`)
  return `UCI Machine Learning Repository dataset ${info.uciId} (${info.files})`
}

function uciMeta(name: string, sha256: string): Partial<DatasetMeta> {
  const info = UCI_INFO[name]
  if (info === undefined) throw new OmaiError('value', `uciMeta: no UCI dataset '${name}'`)
  return {
    name,
    description: info.description,
    url: uciUrl(name),
    citation: info.citation,
    licence: UCI_LICENCE,
    sha256,
    source: uciSource(name),
  }
}

/** A classification dataset from a generated payload. */
export function uciClassification(
  name: string,
  payload: {
    readonly shape: readonly [number, number]
    readonly x: readonly number[]
    readonly y: readonly number[]
    readonly features: readonly string[]
    readonly classes: readonly string[]
    readonly sha256: string
  },
): ClassificationData {
  const [n, d] = payload.shape
  return xyClasses(
    fromStorage(Float64Array.from(payload.x), [n, d], 'float64'),
    fromStorage(Int32Array.from(payload.y), [n], 'int32'),
    { ...uciMeta(name, payload.sha256), featureNames: payload.features, classes: payload.classes },
  )
}

/** A regression dataset from a generated payload (several targets when `targets` has several names). */
export function uciRegression(
  name: string,
  payload: {
    readonly shape: readonly [number, number]
    readonly x: readonly number[]
    readonly y: readonly number[]
    readonly features: readonly string[]
    readonly targets: readonly string[]
    readonly sha256: string
  },
): RegressionData {
  const [n, d] = payload.shape
  const k = payload.targets.length
  return xy(
    fromStorage(Float64Array.from(payload.x), [n, d], 'float64'),
    fromStorage(Float64Array.from(payload.y), k === 1 ? [n] : [n, k], 'float64'),
    { ...uciMeta(name, payload.sha256), featureNames: payload.features, targetNames: payload.targets },
  )
}

/** A table from a generated payload: its target, its text columns and the levels of the categorical columns. */
export function uciTable(
  name: string,
  payload: {
    readonly columns: Readonly<Record<string, readonly (number | null)[] | readonly (string | null)[]>>
    readonly sha256: string
  },
  options: {
    readonly target: string
    readonly text?: readonly string[]
    readonly levels?: Readonly<Record<string, readonly string[]>>
  },
): TableData {
  return table(payload.columns, {
    target: options.target,
    ...(options.text === undefined || options.text.length === 0 ? {} : { text: options.text }),
    ...(options.levels === undefined ? {} : { levels: options.levels }),
    meta: uciMeta(name, payload.sha256),
  })
}
