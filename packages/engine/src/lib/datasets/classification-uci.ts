/**
 * Real numeric classification datasets from the UCI Machine Learning Repository (all CC BY 4.0): banknote
 * authentication, ionosphere, sonar, glass, seeds, E. coli and yeast proteins, Haberman's survival, the balance scale,
 * vehicle silhouettes, wholesale customers, lenses and a subsample of spambase. Each is a vendored payload generated
 * from the published file by `oracle/datasets`.
 */

import type { ClassificationData } from '../data/index.ts'
import {
  BALANCE_SCALE_CLASSES,
  BALANCE_SCALE_FEATURES,
  BALANCE_SCALE_SHA256,
  BALANCE_SCALE_SHAPE,
  BALANCE_SCALE_X,
  BALANCE_SCALE_Y,
} from './_data/balance-scale.data.ts'
import {
  BANKNOTE_CLASSES,
  BANKNOTE_FEATURES,
  BANKNOTE_SHA256,
  BANKNOTE_SHAPE,
  BANKNOTE_X,
  BANKNOTE_Y,
} from './_data/banknote.data.ts'
import { ECOLI_CLASSES, ECOLI_FEATURES, ECOLI_SHA256, ECOLI_SHAPE, ECOLI_X, ECOLI_Y } from './_data/ecoli.data.ts'
import { GLASS_CLASSES, GLASS_FEATURES, GLASS_SHA256, GLASS_SHAPE, GLASS_X, GLASS_Y } from './_data/glass.data.ts'
import {
  HABERMAN_CLASSES,
  HABERMAN_FEATURES,
  HABERMAN_SHA256,
  HABERMAN_SHAPE,
  HABERMAN_X,
  HABERMAN_Y,
} from './_data/haberman.data.ts'
import {
  IONOSPHERE_CLASSES,
  IONOSPHERE_FEATURES,
  IONOSPHERE_SHA256,
  IONOSPHERE_SHAPE,
  IONOSPHERE_X,
  IONOSPHERE_Y,
} from './_data/ionosphere.data.ts'
import {
  LENSES_CLASSES,
  LENSES_FEATURES,
  LENSES_SHA256,
  LENSES_SHAPE,
  LENSES_X,
  LENSES_Y,
} from './_data/lenses.data.ts'
import { SEEDS_CLASSES, SEEDS_FEATURES, SEEDS_SHA256, SEEDS_SHAPE, SEEDS_X, SEEDS_Y } from './_data/seeds.data.ts'
import { SONAR_CLASSES, SONAR_FEATURES, SONAR_SHA256, SONAR_SHAPE, SONAR_X, SONAR_Y } from './_data/sonar.data.ts'
import {
  SPAMBASE_CLASSES,
  SPAMBASE_FEATURES,
  SPAMBASE_SHA256,
  SPAMBASE_SHAPE,
  SPAMBASE_X,
  SPAMBASE_Y,
} from './_data/spambase.data.ts'
import {
  VEHICLE_CLASSES,
  VEHICLE_FEATURES,
  VEHICLE_SHA256,
  VEHICLE_SHAPE,
  VEHICLE_X,
  VEHICLE_Y,
} from './_data/vehicle.data.ts'
import {
  WHOLESALE_CLASSES,
  WHOLESALE_FEATURES,
  WHOLESALE_SHA256,
  WHOLESALE_SHAPE,
  WHOLESALE_X,
  WHOLESALE_Y,
} from './_data/wholesale.data.ts'
import { YEAST_CLASSES, YEAST_FEATURES, YEAST_SHA256, YEAST_SHAPE, YEAST_X, YEAST_Y } from './_data/yeast.data.ts'
import { uciClassification } from './_uci.ts'

/**
 * The UCI banknote authentication data: four wavelet statistics (variance, skewness, curtosis, entropy) of 1372 images
 * of genuine and forged banknotes. The repository does not say which of the two labels, `0` and `1`, is the forged
 * class, so the classes keep the published labels. Embedded from the repository's `data_banknote_authentication.txt`
 * (CC BY 4.0).
 *
 * @returns The dataset: features `x`, class indices `y`, the class names and the feature names.
 * @concept datasets
 * @example
 * ```ts
 * import { banknoteAuthentication } from 'omai-engine/datasets'
 * banknoteAuthentication().x.shape // => [1372, 4]
 * ```
 */
export function banknoteAuthentication(): ClassificationData {
  return uciClassification('banknoteAuthentication', {
    shape: BANKNOTE_SHAPE,
    x: BANKNOTE_X,
    y: BANKNOTE_Y,
    features: BANKNOTE_FEATURES,
    classes: BANKNOTE_CLASSES,
    sha256: BANKNOTE_SHA256,
  })
}

/**
 * The UCI ionosphere data: 351 radar returns, each 17 pulse numbers as a real and an imaginary part (34 features),
 * labelled good (signs of structure in the ionosphere) or bad. The second feature is constant zero, as published.
 * Embedded from the repository's `ionosphere.data` (CC BY 4.0).
 *
 * @returns The dataset: features `x`, class indices `y`, the class names and the feature names.
 * @concept datasets
 * @example
 * ```ts
 * import { ionosphere } from 'omai-engine/datasets'
 * ionosphere().x.shape // => [351, 34]
 * ```
 */
export function ionosphere(): ClassificationData {
  return uciClassification('ionosphere', {
    shape: IONOSPHERE_SHAPE,
    x: IONOSPHERE_X,
    y: IONOSPHERE_Y,
    features: IONOSPHERE_FEATURES,
    classes: IONOSPHERE_CLASSES,
    sha256: IONOSPHERE_SHA256,
  })
}

/**
 * The UCI sonar data: 208 returns of a sonar signal in 60 energy bands, bounced off a metal cylinder (mine) or a rock.
 * Embedded from the repository's `sonar.all-data` (CC BY 4.0).
 *
 * @returns The dataset: features `x`, class indices `y`, the class names and the feature names.
 * @concept datasets
 * @example
 * ```ts
 * import { sonar } from 'omai-engine/datasets'
 * sonar().x.shape // => [208, 60]
 * ```
 */
export function sonar(): ClassificationData {
  return uciClassification('sonar', {
    shape: SONAR_SHAPE,
    x: SONAR_X,
    y: SONAR_Y,
    features: SONAR_FEATURES,
    classes: SONAR_CLASSES,
    sha256: SONAR_SHA256,
  })
}

/**
 * The UCI glass identification data: the refractive index and the weight percent of eight oxides in 214 glass samples
 * of six types (type 4, absent from the data, is skipped). The running id of the file is dropped. Embedded from the
 * repository's `glass.data` (CC BY 4.0).
 *
 * @returns The dataset: features `x`, class indices `y`, the class names and the feature names.
 * @concept datasets
 * @example
 * ```ts
 * import { glassIdentification } from 'omai-engine/datasets'
 * glassIdentification().classes.length // => 6
 * ```
 */
export function glassIdentification(): ClassificationData {
  return uciClassification('glassIdentification', {
    shape: GLASS_SHAPE,
    x: GLASS_X,
    y: GLASS_Y,
    features: GLASS_FEATURES,
    classes: GLASS_CLASSES,
    sha256: GLASS_SHA256,
  })
}

/**
 * The UCI seeds data: seven geometric measurements (from X-ray images) of 210 wheat kernels, 70 each of the varieties
 * Kama, Rosa and Canadian. Embedded from the repository's `seeds_dataset.txt` (CC BY 4.0).
 *
 * @returns The dataset: features `x`, class indices `y`, the class names and the feature names.
 * @concept datasets
 * @example
 * ```ts
 * import { seeds } from 'omai-engine/datasets'
 * seeds().classes // => ['Kama', 'Rosa', 'Canadian']
 * ```
 */
export function seeds(): ClassificationData {
  return uciClassification('seeds', {
    shape: SEEDS_SHAPE,
    x: SEEDS_X,
    y: SEEDS_Y,
    features: SEEDS_FEATURES,
    classes: SEEDS_CLASSES,
    sha256: SEEDS_SHA256,
  })
}

/**
 * The UCI E. coli data: seven scores computed from the amino-acid sequence of 336 proteins, each in one of eight
 * cellular localisation sites (two of them with only two proteins). The protein's sequence name is dropped. Embedded
 * from the repository's `ecoli.data` (CC BY 4.0).
 *
 * @returns The dataset: features `x`, class indices `y`, the class names and the feature names.
 * @concept datasets
 * @example
 * ```ts
 * import { ecoli } from 'omai-engine/datasets'
 * ecoli().x.shape // => [336, 7]
 * ```
 */
export function ecoli(): ClassificationData {
  return uciClassification('ecoli', {
    shape: ECOLI_SHAPE,
    x: ECOLI_X,
    y: ECOLI_Y,
    features: ECOLI_FEATURES,
    classes: ECOLI_CLASSES,
    sha256: ECOLI_SHA256,
  })
}

/**
 * The UCI yeast data: eight scores computed from the amino-acid sequence of 1484 proteins, each in one of ten cellular
 * localisation sites, very unevenly populated. The protein's sequence name is dropped. Embedded from the repository's
 * `yeast.data` (CC BY 4.0).
 *
 * @returns The dataset: features `x`, class indices `y`, the class names and the feature names.
 * @concept datasets
 * @example
 * ```ts
 * import { yeast } from 'omai-engine/datasets'
 * yeast().x.shape // => [1484, 8]
 * ```
 */
export function yeast(): ClassificationData {
  return uciClassification('yeast', {
    shape: YEAST_SHAPE,
    x: YEAST_X,
    y: YEAST_Y,
    features: YEAST_FEATURES,
    classes: YEAST_CLASSES,
    sha256: YEAST_SHA256,
  })
}

/**
 * The UCI Haberman survival data: age, year of operation (19xx, two digits) and the number of positive axillary nodes
 * of 306 patients who had breast-cancer surgery, with whether they survived five years or more. Embedded from the
 * repository's `haberman.data` (CC BY 4.0).
 *
 * @returns The dataset: features `x`, class indices `y`, the class names and the feature names.
 * @concept datasets
 * @example
 * ```ts
 * import { haberman } from 'omai-engine/datasets'
 * haberman().classes // => ['survived', 'died']
 * ```
 */
export function haberman(): ClassificationData {
  return uciClassification('haberman', {
    shape: HABERMAN_SHAPE,
    x: HABERMAN_X,
    y: HABERMAN_Y,
    features: HABERMAN_FEATURES,
    classes: HABERMAN_CLASSES,
    sha256: HABERMAN_SHA256,
  })
}

/**
 * The UCI balance scale data: every combination of a left and right weight and distance (each 1 to 5), labelled by
 * whether the scale tips left, balances or tips right, which is decided by comparing the two products. Embedded from
 * the repository's `balance-scale.data` (CC BY 4.0).
 *
 * @returns The dataset: features `x`, class indices `y`, the class names and the feature names.
 * @concept datasets
 * @example
 * ```ts
 * import { balanceScale } from 'omai-engine/datasets'
 * balanceScale().x.shape // => [625, 4]
 * ```
 */
export function balanceScale(): ClassificationData {
  return uciClassification('balanceScale', {
    shape: BALANCE_SCALE_SHAPE,
    x: BALANCE_SCALE_X,
    y: BALANCE_SCALE_Y,
    features: BALANCE_SCALE_FEATURES,
    classes: BALANCE_SCALE_CLASSES,
    sha256: BALANCE_SCALE_SHA256,
  })
}

/**
 * The UCI Statlog vehicle silhouettes data: 18 shape features of the silhouettes of a double-decker bus, a Chevrolet
 * van, a Saab 9000 and an Opel Manta, seen from different angles. The nine files of the repository are joined in
 * order. Embedded from the repository's `xaa.dat to xai.dat` (CC BY 4.0).
 *
 * @returns The dataset: features `x`, class indices `y`, the class names and the feature names.
 * @concept datasets
 * @example
 * ```ts
 * import { vehicleSilhouettes } from 'omai-engine/datasets'
 * vehicleSilhouettes().classes // => ['bus', 'opel', 'saab', 'van']
 * ```
 */
export function vehicleSilhouettes(): ClassificationData {
  return uciClassification('vehicleSilhouettes', {
    shape: VEHICLE_SHAPE,
    x: VEHICLE_X,
    y: VEHICLE_Y,
    features: VEHICLE_FEATURES,
    classes: VEHICLE_CLASSES,
    sha256: VEHICLE_SHA256,
  })
}

/**
 * The UCI wholesale customers data: the region and the yearly spending (monetary units) on fresh, milk, grocery,
 * frozen, detergents and paper, and delicatessen products of 440 clients of a Portuguese wholesaler. The class is the
 * channel (hotel, restaurant or cafe, or retail); the data are also a standard clustering benchmark. Embedded from the
 * repository's `Wholesale customers data.csv` (CC BY 4.0).
 *
 * @returns The dataset: features `x`, class indices `y`, the class names and the feature names.
 * @concept datasets
 * @example
 * ```ts
 * import { wholesaleCustomers } from 'omai-engine/datasets'
 * wholesaleCustomers().classes // => ['horeca', 'retail']
 * ```
 */
export function wholesaleCustomers(): ClassificationData {
  return uciClassification('wholesaleCustomers', {
    shape: WHOLESALE_SHAPE,
    x: WHOLESALE_X,
    y: WHOLESALE_Y,
    features: WHOLESALE_FEATURES,
    classes: WHOLESALE_CLASSES,
    sha256: WHOLESALE_SHA256,
  })
}

/**
 * The UCI lenses data, a decision-tree toy: all 24 combinations of the patient's age, spectacle prescription,
 * astigmatism and tear production rate, with the fitting decision. The attributes keep their published integer codes
 * (age 1 young to 3 presbyopic; prescription 1 myope, 2 hypermetrope; astigmatic 1 no, 2 yes; tear rate 1 reduced, 2
 * normal). Embedded from the repository's `lenses.data` (CC BY 4.0).
 *
 * @returns The dataset: features `x`, class indices `y`, the class names and the feature names.
 * @concept datasets
 * @example
 * ```ts
 * import { lenses } from 'omai-engine/datasets'
 * lenses().x.shape // => [24, 4]
 * ```
 */
export function lenses(): ClassificationData {
  return uciClassification('lenses', {
    shape: LENSES_SHAPE,
    x: LENSES_X,
    y: LENSES_Y,
    features: LENSES_FEATURES,
    classes: LENSES_CLASSES,
    sha256: LENSES_SHA256,
  })
}

/**
 * The UCI Spambase data as a stratified subsample of 1000 of its 4601 e-mails (the full file is too large to vendor):
 * 57 features (word and character frequencies in percent, and capital-letter run lengths) and whether the e-mail is
 * spam. The sample keeps the spam share (394 of 1000, as 1813 of 4601) and the original row order; the rows are chosen
 * by the SHA-256 of their index, so the sample is reproducible. Embedded from the repository's `spambase.data` (CC BY
 * 4.0).
 *
 * @returns The dataset: features `x`, class indices `y`, the class names and the feature names.
 * @concept datasets
 * @example
 * ```ts
 * import { spambase } from 'omai-engine/datasets'
 * spambase().x.shape // => [1000, 57]
 * ```
 */
export function spambase(): ClassificationData {
  return uciClassification('spambase', {
    shape: SPAMBASE_SHAPE,
    x: SPAMBASE_X,
    y: SPAMBASE_Y,
    features: SPAMBASE_FEATURES,
    classes: SPAMBASE_CLASSES,
    sha256: SPAMBASE_SHA256,
  })
}
