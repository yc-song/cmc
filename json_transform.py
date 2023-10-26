import json
from tqdm import tqdm

cands = []
with open('/shared/s3/lab07/jongsong/hard-nce-el/data/cands_anncur_top1024_copy/candidates_train_w_reranker_score.json', 'r') as f:
    for i, line in tqdm(enumerate(f)):
        try:
            cands.append(json.loads(line))
        except:
            print(i)
for i, cand in enumerate(cands):
    try: 
        cand["scores"] = cand["reranker_scores"]
        del(cand["reranker_scores"])
    except: print(i)
with open('/shared/s3/lab07/jongsong/hard-nce-el/data/cands_anncur_top1024_copy/candidates_train_w.json', 'w') as f:
    for item in tqdm(cands):
        try:
            f.write('%s\n' % json.dumps(item))
        except:
            print(item)