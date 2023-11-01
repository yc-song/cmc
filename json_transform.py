import json
from tqdm import tqdm

# cands = []
# new_cands = []
# with open('/shared/s3/lab07/jongsong/hard-nce-el/data/backup/candidates_train.json', 'r') as f:
#     for i, line in tqdm(enumerate(f)):
#         cands.append(dict(eval(line)))
#     print(len(cands))
# for item in tqdm(cands[10500:10600]):
#     new_cands.append(item)
# for item in tqdm(cands[:10500]):
#     new_cands.append(item)
# for item in tqdm(cands[10600:]):
#     new_cands.append(item)
# with open('/shared/s3/lab07/jongsong/hard-nce-el/data/backup/candidates_train_debug.json', 'w') as f:
#     for item in tqdm(new_cands):
#         f.write('%s\n' % json.dumps(dict(item)))


cands = []
new_cands = []
with open('/shared/s3/lab07/jongsong/hard-nce-el/data/backup/candidates_train.json', 'r') as f:
    for i, line in tqdm(enumerate(f)):
        if 10500<i and i<10600:
            cands.append(dict(eval(line)))
with open('/shared/s3/lab07/jongsong/hard-nce-el/data/backup/candidates_train_debug.json', 'w') as f:
    for item in tqdm(cands):
        f.write('%s\n' % json.dumps(dict(item)))