/**
 * Small real classification datasets vendored into the bundle: Fisher's iris, the UCI wine and Wisconsin breast cancer
 * data, the 8 by 8 handwritten digits (as features and as images), the Palmer penguins and the Titanic survival table,
 * as in scikit-learn's `load_*` functions.
 */

import {
  type ClassificationData,
  type ImageData,
  images,
  type TableColumn,
  type TableData,
  table,
  tableToXy,
  xyClasses,
} from '../data/index.ts'
import { fromStorage } from '../tensor/index.ts'
import {
  BREAST_CANCER_CLASSES,
  BREAST_CANCER_FEATURES,
  BREAST_CANCER_SHA256,
  BREAST_CANCER_SHAPE,
  BREAST_CANCER_X,
  BREAST_CANCER_Y,
} from './_data/breast-cancer.data.ts'
import { DIGITS_LABELS, DIGITS_PIXELS, DIGITS_SHA256, DIGITS_SHAPE } from './_data/digits.data.ts'
import { IRIS_CLASSES, IRIS_FEATURES, IRIS_SHA256, IRIS_SHAPE, IRIS_X, IRIS_Y } from './_data/iris.data.ts'
import { PENGUINS_COLUMNS, PENGUINS_SHA256 } from './_data/penguins.data.ts'
import { TITANIC_COLUMNS, TITANIC_SHA256 } from './_data/titanic.data.ts'
import { WINE_CLASSES, WINE_FEATURES, WINE_SHA256, WINE_SHAPE, WINE_X, WINE_Y } from './_data/wine.data.ts'

const SKLEARN_LICENCE = 'BSD-3-Clause (scikit-learn); the data are from the UCI repository (CC BY 4.0)'

function numericClassification(
  shape: readonly [number, number],
  x: readonly number[],
  y: readonly number[],
  features: readonly string[],
  classes: readonly string[],
  meta: { name: string; description: string; url: string; citation: string; sha256: string; source: string },
): ClassificationData {
  const [n, d] = shape
  return xyClasses(
    fromStorage(Float64Array.from(x), [n, d], 'float64'),
    fromStorage(Int32Array.from(y), [n], 'int32'),
    { ...meta, licence: SKLEARN_LICENCE, featureNames: features, classes },
  )
}

/**
 * Fisher's iris data: four measurements (sepal and petal length and width, in cm) of 150 irises of three species.
 * Embedded from scikit-learn's `load_iris`.
 *
 * @returns The dataset.
 * @concept datasets
 * @example
 * ```ts
 * import { iris } from 'whitebox-engine/datasets'
 * iris().classes // => ['setosa', 'versicolor', 'virginica']
 * ```
 */
export function iris(): ClassificationData {
  return numericClassification(IRIS_SHAPE, IRIS_X, IRIS_Y, IRIS_FEATURES, IRIS_CLASSES, {
    name: 'iris',
    description: 'Sepal and petal measurements of 150 irises of three species (Fisher, 1936).',
    url: 'https://archive.ics.uci.edu/dataset/53/iris',
    citation: 'Fisher (1936), The use of multiple measurements in taxonomic problems, Annals of Eugenics 7(2).',
    sha256: IRIS_SHA256,
    source: 'scikit-learn load_iris',
  })
}

/**
 * The UCI wine recognition data: 13 chemical measurements of 178 Italian wines from three cultivars. Embedded from
 * scikit-learn's `load_wine`.
 *
 * @returns The dataset.
 * @example
 * ```ts
 * import { wine } from 'whitebox-engine/datasets'
 * wine().x.shape // => [178, 13]
 * ```
 */
export function wine(): ClassificationData {
  return numericClassification(WINE_SHAPE, WINE_X, WINE_Y, WINE_FEATURES, WINE_CLASSES, {
    name: 'wine',
    description: 'Chemical analysis of 178 wines from three cultivars in the same region of Italy.',
    url: 'https://archive.ics.uci.edu/dataset/109/wine',
    citation: 'Aeberhard and Forina (1991), Wine, UCI Machine Learning Repository.',
    sha256: WINE_SHA256,
    source: 'scikit-learn load_wine',
  })
}

/**
 * The Wisconsin diagnostic breast cancer data: 30 features of cell nuclei from 569 biopsies, malignant (0) or
 * benign (1). Embedded from scikit-learn's `load_breast_cancer`.
 *
 * @returns The dataset.
 * @example
 * ```ts
 * import { breastCancer } from 'whitebox-engine/datasets'
 * breastCancer().classes // => ['malignant', 'benign']
 * ```
 */
export function breastCancer(): ClassificationData {
  return numericClassification(
    BREAST_CANCER_SHAPE,
    BREAST_CANCER_X,
    BREAST_CANCER_Y,
    BREAST_CANCER_FEATURES,
    BREAST_CANCER_CLASSES,
    {
      name: 'breastCancer',
      description: 'Features of cell nuclei in 569 breast masses, malignant or benign (Street et al., 1993).',
      url: 'https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic',
      citation: 'Street, Wolberg and Mangasarian (1993), Nuclear feature extraction for breast tumor diagnosis.',
      sha256: BREAST_CANCER_SHA256,
      source: 'scikit-learn load_breast_cancer',
    },
  )
}

function digitPixels(): { pixels: Float64Array; labels: Int32Array } {
  const pixels = Float64Array.from(DIGITS_PIXELS, (ch) => Number.parseInt(ch, 17))
  const labels = Int32Array.from(DIGITS_LABELS, (ch) => Number(ch))
  return { pixels, labels }
}

const DIGITS_META = {
  name: 'digits',
  description: '1797 handwritten digits as 8 by 8 images with grey levels 0 to 16 (UCI optical recognition).',
  url: 'https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits',
  citation: 'Alpaydin and Kaynak (1998), Optical Recognition of Handwritten Digits, UCI Machine Learning Repository.',
  sha256: DIGITS_SHA256,
  source: 'scikit-learn load_digits',
  licence: SKLEARN_LICENCE,
}

/**
 * The 8 by 8 handwritten digits as a classification dataset: 64 pixel features with grey levels 0 to 16. Embedded
 * from scikit-learn's `load_digits`; {@link digitImages} gives the same data as images.
 *
 * @returns The dataset.
 * @example
 * ```ts
 * import { digits } from 'whitebox-engine/datasets'
 * digits().x.shape // => [1797, 64]
 * ```
 */
export function digits(): ClassificationData {
  const [n, d] = DIGITS_SHAPE
  const { pixels, labels } = digitPixels()
  return xyClasses(fromStorage(pixels, [n, d], 'float64'), fromStorage(labels, [n], 'int32'), {
    ...DIGITS_META,
    featureNames: Array.from({ length: d }, (_, i) => `pixel_${Math.floor(i / 8)}_${i % 8}`),
    classes: Array.from({ length: 10 }, (_, i) => String(i)),
  })
}

/**
 * The 8 by 8 handwritten digits as images `[1797, 8, 8, 1]`, scaled to [0, 1] (grey level / 16).
 *
 * @returns The image dataset.
 * @example
 * ```ts
 * import { digitImages } from 'whitebox-engine/datasets'
 * digitImages().images.shape // => [1797, 8, 8, 1]
 * ```
 */
export function digitImages(): ImageData {
  const [n] = DIGITS_SHAPE
  const { pixels, labels } = digitPixels()
  return images(
    fromStorage(
      Float64Array.from(pixels, (v) => v / 16),
      [n, 8, 8, 1],
      'float64',
    ),
    {
      labels: fromStorage(labels, [n], 'int32'),
      classes: Array.from({ length: 10 }, (_, i) => String(i)),
      meta: DIGITS_META,
    },
  )
}

/**
 * The Palmer penguins: species, island, bill length and depth, flipper length, body mass, sex and year of 344
 * penguins (Gorman, Williams and Fraser, 2014), as a table with missing values and the species as target. The
 * licence is CC0.
 *
 * @returns The table.
 * @example
 * ```ts
 * import { penguins } from 'whitebox-engine/datasets'
 * penguins().columns['species']?.kind // => 'categorical'
 * ```
 */
export function penguins(): TableData {
  return table(PENGUINS_COLUMNS as Readonly<Record<string, readonly (number | null)[] | readonly (string | null)[]>>, {
    target: 'species',
    meta: {
      name: 'penguins',
      description: 'Measurements of 344 Adelie, Chinstrap and Gentoo penguins from three islands near Palmer Station.',
      url: 'https://allisonhorst.github.io/palmerpenguins/',
      citation:
        'Gorman, Williams and Fraser (2014), Ecological sexual dimorphism and environmental variability within a community of Antarctic penguins, PLoS ONE 9(3).',
      licence: 'CC0-1.0',
      sha256: PENGUINS_SHA256,
      source: 'palmerpenguins penguins.csv',
    },
  })
}

/**
 * The Palmer penguins as a numeric classification dataset: the four measurements of the 342 penguins that have
 * them, with the species as class.
 *
 * @returns The dataset.
 * @example
 * ```ts
 * import { penguinsClassification } from 'whitebox-engine/datasets'
 * penguinsClassification().x.shape // => [342, 4]
 * ```
 */
export function penguinsClassification(): ClassificationData {
  const t = penguins()
  const keep = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'species']
  const columns: Record<string, TableColumn> = {}
  for (const name of keep) {
    const c = t.columns[name]
    if (c !== undefined) columns[name] = c
  }
  const d = tableToXy({ ...t, columns }) as ClassificationData
  const rows: number[] = []
  const [n = 0, w = 1] = d.x.shape
  for (let i = 0; i < n; i++) {
    let ok = true
    for (let j = 0; j < w; j++) if (Number.isNaN(d.x.data[i * w + j] ?? Number.NaN)) ok = false
    if (ok) rows.push(i)
  }
  const x = Float64Array.from(rows.flatMap((i) => Array.from(d.x.data.slice(i * w, (i + 1) * w))))
  const y = Int32Array.from(rows, (i) => d.y.data[i] ?? 0)
  return xyClasses(fromStorage(x, [rows.length, w], 'float64'), fromStorage(y, [rows.length], 'int32'), {
    ...t.meta,
    name: 'penguinsClassification',
    featureNames: d.featureNames,
    classes: d.classes,
  })
}

/**
 * Survival of the 2201 people aboard the Titanic by class (1st, 2nd, 3rd, crew), sex and age (child, adult), the
 * table of R's `datasets::Titanic` (from the British Board of Trade inquiry, 1912), expanded to one row per person.
 * The target is `survived`.
 *
 * @returns The table.
 * @example
 * ```ts
 * import { titanic } from 'whitebox-engine/datasets'
 * import { rowCount } from 'whitebox-engine/data'
 * rowCount(titanic()) // => 2201
 * ```
 */
export function titanic(): TableData {
  const freq = TITANIC_COLUMNS['Freq'] as readonly number[]
  const col = (name: string) => TITANIC_COLUMNS[name] as readonly string[]
  const out: Record<string, string[]> = { class: [], sex: [], age: [], survived: [] }
  for (const [i, f] of freq.entries()) {
    for (let r = 0; r < f; r++) {
      out['class']?.push(col('Class')[i] ?? '')
      out['sex']?.push((col('Sex')[i] ?? '').toLowerCase())
      out['age']?.push((col('Age')[i] ?? '').toLowerCase())
      out['survived']?.push((col('Survived')[i] ?? '').toLowerCase())
    }
  }
  return table(out, {
    target: 'survived',
    levels: {
      class: ['1st', '2nd', '3rd', 'Crew'],
      sex: ['female', 'male'],
      age: ['child', 'adult'],
      survived: ['no', 'yes'],
    },
    meta: {
      name: 'titanic',
      description: 'Class, sex, age and survival of the 2201 people aboard the Titanic (R datasets::Titanic).',
      url: 'https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/Titanic.html',
      citation: 'Dawson (1995), The unusual episode data revisited, Journal of Statistics Education 3(3).',
      licence: 'Public domain (British Board of Trade, 1912); table from R (GPL-2 | GPL-3)',
      sha256: TITANIC_SHA256,
      source: 'R datasets::Titanic via Rdatasets',
    },
  })
}
