import json
import os
mrr = 0.
count = 0
cands = []
with open('/shared/s3/lab07/jongsong/hard-nce-el/data/candidates_train_w_reranker_score.json', 'r') as f:
    for line in f:
        cands.append(json.loads(line))
for item in cands:
    try:
        item['scores'] = item['reranker_scores']
        del item['reranker_scores']
    except KeyError:
        pass
with open('/shared/s3/lab07/jongsong/hard-nce-el/data/candidates_train_w_reranker_score_2.json', 'w') as f:
    for item in cands:
        f.write('%s\n' % json.dumps(item))
