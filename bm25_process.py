import json
mode = 'train'
lines = []
with open(f'./data/mentions/{mode}.json') as f2:
    for line in f2:
        lines.append(eval(line.replace('\n', '')))
cands = []
with open(f'./data/bm25/{mode}.json', 'r') as f1:
    for i, line in enumerate(f1):
        cand = {}
        line = eval(line)
        cand['mention_id'] = line['mention_id']
        cand['candidates'] = line['tfidf_candidates']
        cand['scores'] = [0 for i in cand['candidates']]
        # if mode in ['val', 'test']:
        #     try:
        #         cand['labels'] = cand['candidates'].index(lines[i]['label_document_id'])
        #         cands.append(cand)
        #     except:
        #         pass
        try:
            cand['labels'] = cand['candidates'].index(lines[i]['label_document_id'])
        except:
            cand['labels'] = -1
        cands.append(cand)

with open(f'./data/bm25/candidates_{mode}.json', 'w') as f:
    for cand in cands:
        json_line = json.dumps(cand)
        f.write(json_line + '\n')