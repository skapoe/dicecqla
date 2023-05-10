from dicee import KGE
import numpy as np
import pickle
import os


def load_data(data_path,tasks):

        #load queries from the datapath
        queries = pickle.load(open(os.path.join(data_path, "test-queries.pkl"), 'rb'))
        answers_hard = pickle.load(open(os.path.join(data_path, "test-hard-answers.pkl"), 'rb'))
        answers_easy = pickle.load(open(os.path.join(data_path, "test-easy-answers.pkl"), 'rb'))

        for task in list(queries.keys()):
            if task not in query_name_dict or query_name_dict[task] not in tasks:
                del queries[task]
        # for qs in tasks:
        #     try:
        #         logging.info(_type + ': ' + qs + ": " + str(len(queries[name_query_dict[qs]])))
        #     except:
        #         logging.warn(_type + ': ' + qs + ": not in pkl file")

        return queries, answers_easy, answers_hard



query_name_dict = {

        ("e", ("r",)): "1p",
        ("e", ("r", "r")): "2p",
        ("e", ("r", "r", "r",),): "3p",
        (("e", ("r",)), ("e", ("r",))): "2i",
        (("e", ("r",)), ("e", ("r",)), ("e", ("r",))): "3i",
        ((("e", ("r",)), ("e", ("r",))), ("r",)): "ip",
        (("e", ("r", "r")), ("e", ("r",))): "pi",
        # negation
        (("e", ("r",)), ("e", ("r", "n"))): "2in",
        (("e", ("r",)), ("e", ("r",)), ("e", ("r", "n"))): "3in",
        ((("e", ("r",)), ("e", ("r", "n"))), ("r",)): "inp",
        (("e", ("r", "r")), ("e", ("r", "n"))): "pin",
        (("e", ("r", "r", "n")), ("e", ("r",))): "pni",

        # union
        (("e", ("r",)), ("e", ("r",)), ("u",)): "2u",
        ((("e", ("r",)), ("e", ("r",)), ("u",)), ("r",)): "up",

    }
name_query_dict = {value: key for key, value in query_name_dict.items()}
def scores_2in(model, queries):
    # Function to calculate entity scores for type 2in structure

    entity_scores = {}

    for query in queries:
        head1, relation1 = query[0]
        head2, relation2 = query[1]

        # Calculate entity scores for each query
        # Get scores for the first atom (positive)
        atom1_scores = model.predict(head_entities=[head1], relations=[relation1[0]]).squeeze()
        # Get scores for the second atom (negative)
        # modelling standard negation (1-x)
        atom2_scores = 1 - model.predict(head_entities=[head2], relations=[relation2[0]]).squeeze()

        assert len(atom1_scores) == len(model.entity_to_idx)
        combined_scores = []

        # Combine scores from both atoms
        # modelling T-min norm Tnorm=min(a,b)
        for ei, s1, s2 in zip(model.entity_to_idx.keys(), atom1_scores, atom2_scores):
            if s1 > s2:
                combined_scores.append((ei, float(s2)))
            else:
                combined_scores.append((ei, float(s1)))

        # Sort combined scores in descending order
        combined_scores = sorted(combined_scores, key=lambda x: x[1], reverse=True)
        entity_scores[query] = combined_scores

    return entity_scores


def scores_2i(model, queries):
    # Function to calculate entity scores for type 2i structure

    entity_scores = {}

    for query in queries:
        head1, relation1 = query[0]
        head2, relation2 = query[1]

        # Calculate entity scores for each query
        # Get scores for the first atom
        atom1_scores = model.predict(head_entities=[head1], relations=[relation1[0]]).squeeze()
        # Get scores for the second atom
        atom2_scores = model.predict(head_entities=[head2], relations=[relation2[0]]).squeeze()

        assert len(atom1_scores) == len(model.entity_to_idx)
        combined_scores = []

        # Combine scores from both atoms
        #modelling T-min norm Tnorm=min(a,b)
        for ei, s1, s2 in zip(model.entity_to_idx.keys(), atom1_scores, atom2_scores):
            if s1 > s2:
                combined_scores.append((ei, float(s2)))
            else:
                combined_scores.append((ei, float(s1)))

        # Sort combined scores in descending order
        combined_scores = sorted(combined_scores, key=lambda x: x[1], reverse=True)
        entity_scores[query] = combined_scores

    return entity_scores

def calculate_mrr(scores, easy_answers, hard_answers):
    # Calculate MRR considering the hard and easy answers
    total_mrr = 0
    total_h1 = 0
    total_h3 = 0
    total_h10 = 0
    num_queries = len(scores)

    for query,entity_scores in scores.items():

        # Extract corresponding easy and hard answers
        easy_ans=easy_answers[query]
        hard_ans=hard_answers[query]
        easy_answer_indices = [idx for idx, (entity, _) in enumerate(entity_scores) if entity in easy_ans]
        hard_answer_indices = [idx for idx, (entity, _) in enumerate(entity_scores) if entity in hard_ans]

        answer_indices = easy_answer_indices + hard_answer_indices

        # The entity_scores list is already sorted
        cur_ranking = np.array(answer_indices)

        # Sort by position in the ranking; indices for (easy + hard) answers
        cur_ranking, indices = np.sort(cur_ranking), np.argsort(cur_ranking)
        num_easy = len(easy_ans)
        num_hard = len(hard_ans)

        # Indices with hard answers only
        masks = indices >= num_easy

        # Reduce ranking for each answer entity by the amount of (easy+hard) answers appearing before it
        answer_list = np.arange(num_hard + num_easy, dtype=float)
        cur_ranking = cur_ranking - answer_list + 1

        # Only take indices that belong to the hard answers
        cur_ranking = cur_ranking[masks]
        # print(cur_ranking)
        mrr = np.mean(1.0 / cur_ranking)
        h1 = np.mean((cur_ranking <= 1).astype(float))
        h3 = np.mean((cur_ranking <= 3).astype(float))
        h10 = np.mean((cur_ranking <= 10).astype(float))
        total_mrr += mrr
        total_h1 += h1
        total_h3 += h3
        total_h10 += h10

    avg_mrr = total_mrr / num_queries
    avg_h1 = total_h1 / num_queries
    avg_h3 = total_h3 / num_queries
    avg_h10 = total_h10 / num_queries

    return avg_mrr, avg_h1, avg_h3, avg_h10


# Add more functions for other types of query structures
def main():
    model = KGE("Experiments/2023-04-29 17:42:37.201428")
    data_path="/Users/sourabh/dice-embeddings/KGs/UMLS"
    tasks = (
                    #"1p",
                    # "2p",
                    # "3p",
                    "2i",
                    # "3i",
                    # "ip",
                    # "pi",
                    "2in",
                    #"3in",
                    #"pin",
                    #"pni",
                    #"inp",
                    # "2u",
                    # "up",
                )

    queries, easy_answers, hard_answers = load_data(data_path, tasks)


    for query_structure, query in queries.items():
        if query_structure == (('e', ('r',)), ('e', ('r', 'n'))):
            entity_scores = scores_2in(model,query)
            mrr, h1, h3, h10 = calculate_mrr(entity_scores, easy_answers, hard_answers)
            print(f"{query_structure}: MRR={mrr}, H1={h1}, H3={h3}, H10={h10}")
        elif query_structure == (('e', ('r',)), ('e', ('r',))):
            entity_scores = scores_2i(model,query)
            mrr, h1, h3, h10 = calculate_mrr(entity_scores, easy_answers, hard_answers)
            print(f"{query_structure}: MRR={mrr}, H1={h1}, H3={h3}, H10={h10}")
        # Add more conditions for other types of query structures



if __name__ == '__main__':
    main()