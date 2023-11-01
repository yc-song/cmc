import json
import os
mrr = 0.
count = 0
with open('/shared/s3/lab07/jongsong/hard-nce-el/data/cands_anncur_top1024/candidates_val.json', 'r') as f:
    for line in f:
        count += 1
        labels=json.loads(line)['labels']
        if labels<512 and labels>=0:
            mrr += 1/(labels+1)

print(mrr/count)