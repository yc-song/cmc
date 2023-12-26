import json
mode = 'val'
cand_all = []
cand_retrieved = []

with open(f'./data/cands_anncur_top1024/5iew1xhb/candidates_{mode}.json', 'r') as f1:
    for i, line in enumerate(f1):
        line = eval(line)
        cand_retrieved.append(line['mention_id'])
with open(f'./data/mentions/{mode}.json', 'r') as f2:
    for line in f2:
        cand_all.append(eval(line.replace('\n', '')))

# print(type(cand_retrieved))
with open(f'./data/cands_anncur_top1024/5iew1xhb/candidates_{mode}.json', 'a') as f1:
    for i, cand in enumerate(cand_all):
        if not cand['mention_id'] in cand_retrieved:
            new_cand ={}
            new_cand['mention_id'] = cand['mention_id']
            new_cand['candidates'] = [0 for i in range(64)]
            new_cand['scores'] = [0 for i in range(64)]
            new_cand['lables'] = -1
            json_line = json.dumps(new_cand)
            f1.write(json_line + '\n')


        # if mode in ['val', 'test']:
        #     try:
        #         cand['labels'] = cand['candidates'].index(lines[i]['label_document_id'])
        #         cands.append(cand)
        #     except:
        #         pass
#         try:
#             cand['labels'] = cand['candidates'].index(lines[i]['label_document_id'])
#         except:
#             cand['labels'] = -1
#         cands.append(cand)

# with open(f'./data/bm25/candidates_{mode}.json', 'w') as f:
#     for cand in cands:
#         json_line = json.dumps(cand)
#         f.write(json_line + '\n')