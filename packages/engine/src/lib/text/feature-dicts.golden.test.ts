import { learnGolden, num, str } from '../../testing/learn.ts'
import { toDense } from '../numerics/sparse/index.ts'
import { dictTransform, dictVectorize, type FeatureRecord, featureHash, type HashedSample } from './feature-dicts.ts'

learnGolden('DictVectorizer and FeatureHasher against scikit-learn', 'text/feature_dicts', {
  dict: (_inputs, p) => {
    const { vectorizer, matrix } = dictVectorize(p['records'] as FeatureRecord[], {
      sort: p['sort'] === true,
      separator: str(p, 'separator'),
    })
    return {
      matrix: toDense(matrix),
      names: vectorizer.featureNames.join('|'),
      transformed: toDense(dictTransform(vectorizer, p['new'] as FeatureRecord[])),
    }
  },
  hash: (_inputs, p) => ({
    matrix: toDense(
      featureHash(p['samples'] as HashedSample[], {
        nFeatures: num(p, 'width'),
        alternateSign: p['alternate'] === true,
      }),
    ),
  }),
})
