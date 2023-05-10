from dicee import KGE
from pprint import pprint
model = KGE('Experiments/2023-04-29 17:42:37.201428')

# x isa	entity AND NOT(x is spatial_concept)
atom1_scores = model.predict(relations=['isa'], tail_entities=['entity'])
atom2_scores = 1-model.predict(relations=['isa'], tail_entities=['spatial_concept'])

assert len(atom1_scores) == len(model.entity_to_idx)
entity_scores = []
for ei, s1, s2 in zip(model.entity_to_idx.keys(), atom1_scores, atom2_scores):
    if s1 > s2:
        entity_scores.append((ei, float(s2)))
    else:
        entity_scores.append((ei, float(s1)))
entity_scores = sorted(entity_scores, key=lambda x: x[1], reverse=True)

pprint(entity_scores[:10])

print(entity_scores)
