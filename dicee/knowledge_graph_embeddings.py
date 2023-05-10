import os
import time
from typing import List, Tuple, Set, Iterable, Dict
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
from .abstracts import BaseInteractiveKGE
from .dataset_classes import TriplePredictionDataset
from .static_funcs import random_prediction, deploy_triple_prediction, deploy_tail_entity_prediction, \
    deploy_relation_prediction, deploy_head_entity_prediction
import numpy as np
import sys


class KGE(BaseInteractiveKGE):
    """ Knowledge Graph Embedding Class for interactive usage of pre-trained models"""

    # @TODO: we can download the model if it is not present locally
    def __init__(self, path, construct_ensemble=False,
                 model_name=None,
                 apply_semantic_constraint=False):
        super().__init__(path=path, construct_ensemble=construct_ensemble, model_name=model_name,
                         apply_semantic_constraint=apply_semantic_constraint)

    def __str__(self):
        return 'KGE | ' + str(self.model)

    def predict_missing_head_entity(self, relation: List[str], tail_entity: List[str]) -> Tuple:
        """
        Given a relation and a tail entity, return top k ranked head entity.

        argmax_{e \in E } f(e,r,t), where r \in R, t \in E.

        Parameter
        ---------
        relation: List[str]

        String representation of selected relations.

        tail_entity: List[str]

        String representation of selected entities.


        k: int

        Highest ranked k entities.

        Returns: Tuple
        ---------

        Highest K scores and entities
        """

        head_entity = torch.arange(0, len(self.entity_to_idx))
        relation = torch.LongTensor([self.relation_to_idx[i] for i in relation])
        tail_entity = torch.LongTensor([self.entity_to_idx[i] for i in tail_entity])
        x = torch.stack((head_entity,
                         relation.repeat(self.num_entities, ),
                         tail_entity.repeat(self.num_entities, )), dim=1)
        return self.model.forward(x)


    def predict_missing_relations(self, head_entity: List[str], tail_entity: List[str]) -> Tuple:
        """
        Given a head entity and a tail entity, return top k ranked relations.

        argmax_{r \in R } f(h,r,t), where h, t \in E.


        Parameter
        ---------
        head_entity: List[str]

        String representation of selected entities.

        tail_entity: List[str]

        String representation of selected entities.


        k: int

        Highest ranked k entities.

        Returns: Tuple
        ---------

        Highest K scores and entities
        """

        head_entity = torch.LongTensor([self.entity_to_idx[i] for i in head_entity])
        relation = torch.arange(0, len(self.relation_to_idx))
        tail_entity = torch.LongTensor([self.entity_to_idx[i] for i in tail_entity])

        x = torch.stack((head_entity.repeat(self.num_relations, ),
                         relation,
                         tail_entity.repeat(self.num_relations, )), dim=1)
        return self.model(x)
        # scores = self.model(x)
        # sort_scores, sort_idxs = torch.topk(scores, topk)
        # return sort_scores, [self.idx_to_relations[i] for i in sort_idxs.tolist()]

    def predict_missing_tail_entity(self, head_entity: List[str], relation: List[str]) -> torch.FloatTensor:
        """
        Given a head entity and a relation, return top k ranked entities

        argmax_{e \in E } f(h,r,e), where h \in E and r \in R.


        Parameter
        ---------
        head_entity: List[str]

        String representation of selected entities.

        tail_entity: List[str]

        String representation of selected entities.

        Returns: Tuple
        ---------

        scores
        """
        x=torch.cat((torch.LongTensor([self.entity_to_idx[i] for i in head_entity]).unsqueeze(-1),
                     torch.LongTensor([self.relation_to_idx[i] for i in relation]).unsqueeze(-1)), dim=1)
        return self.model.forward(x)

    def predict(self, *, head_entities: List[str] = None, relations: List[str] = None, tail_entities: List[str] = None):
        # (1) Sanity checking.
        if head_entities is not None:
            assert isinstance(head_entities, list)
            assert isinstance(head_entities[0], str)
        if relations is not None:
            assert isinstance(relations, list)
            assert isinstance(relations[0], str)
        if tail_entities is not None:
            assert isinstance(tail_entities, list)
            assert isinstance(tail_entities[0], str)
        # (2) Predict missing head entity given a relation and a tail entity.
        if head_entities is None:
            assert relations is not None
            assert tail_entities is not None
            # ? r, t
            scores = self.predict_missing_head_entity(relations, tail_entities)
        # (3) Predict missing relation given a head entity and a tail entity.
        elif relations is None:
            assert head_entities is not None
            assert tail_entities is not None
            # h ? t
            scores = self.__predict_missing_relations(head_entities, tail_entities)
        # (4) Predict missing tail entity given a head entity and a relation
        elif tail_entities is None:
            assert head_entities is not None
            assert relations is not None
            # h r ?
            scores = self.predict_missing_tail_entity(head_entities, relations)
        else:
            assert len(head_entities) == len(relations) == len(tail_entities)
            scores = self.triple_score(head_entities, relations, tail_entities)
        return torch.sigmoid(scores)

    def predict_topk(self, *, head_entity: List[str] = None, relation: List[str] = None, tail_entity: List[str] = None,
                     topk: int = 10):
        """
        Predict missing item in a given triple.



        Parameter
        ---------
        head_entity: List[str]

        String representation of selected entities.

        relation: List[str]

        String representation of selected relations.

        tail_entity: List[str]

        String representation of selected entities.


        k: int

        Highest ranked k item.

        Returns: Tuple
        ---------

        Highest K scores and items
        """

        # (1) Sanity checking.
        if head_entity is not None:
            assert isinstance(head_entity, list)
        if relation is not None:
            assert isinstance(relation, list)
        if tail_entity is not None:
            assert isinstance(tail_entity, list)
        # (2) Predict missing head entity given a relation and a tail entity.
        if head_entity is None:
            assert relation is not None
            assert tail_entity is not None
            # ? r, t
            scores = self.predict_missing_head_entity(relation, tail_entity).flatten()
            if self.apply_semantic_constraint:
                # filter the scores
                for th, i in enumerate(relation):
                    scores[self.domain_constraints_per_rel[self.relation_to_idx[i]]] = -torch.inf

            sort_scores, sort_idxs = torch.topk(scores, topk)
            return torch.sigmoid(sort_scores), [self.idx_to_entity[i] for i in sort_idxs.tolist()]
        # (3) Predict missing relation given a head entity and a tail entity.
        elif relation is None:
            assert head_entity is not None
            assert tail_entity is not None
            # h ? t
            scores = self.predict_missing_relations(head_entity, tail_entity).flatten()
            sort_scores, sort_idxs = torch.topk(scores, topk)
            return torch.sigmoid(sort_scores), [self.idx_to_relations[i] for i in sort_idxs.tolist()]
        # (4) Predict missing tail entity given a head entity and a relation
        elif tail_entity is None:
            assert head_entity is not None
            assert relation is not None
            # h r ?t
            scores = self.predict_missing_tail_entity(head_entity, relation).flatten()
            if self.apply_semantic_constraint:
                # filter the scores
                for th, i in enumerate(relation):
                    scores[self.range_constraints_per_rel[self.relation_to_idx[i]]] = -torch.inf
            sort_scores, sort_idxs = torch.topk(scores, topk)
            return torch.sigmoid(sort_scores), [self.idx_to_entity[i] for i in sort_idxs.tolist()]
        else:
            raise AttributeError('Use triple_score method')

    def triple_score(self, head_entity: List[str] = None, relation: List[str] = None,
                     tail_entity: List[str] = None, logits=False) -> torch.FloatTensor:
        """
        Predict triple score

        Parameter
        ---------
        head_entity: List[str]

        String representation of selected entities.

        relation: List[str]

        String representation of selected relations.

        tail_entity: List[str]

        String representation of selected entities.

        logits: bool

        If logits is True, unnormalized score returned

        Returns: Tuple
        ---------

        pytorch tensor of triple score
        """
        head_entity = torch.LongTensor([self.entity_to_idx[i] for i in head_entity]).reshape(len(head_entity), 1)
        relation = torch.LongTensor([self.relation_to_idx[i] for i in relation]).reshape(len(relation), 1)
        tail_entity = torch.LongTensor([self.entity_to_idx[i] for i in tail_entity]).reshape(len(tail_entity), 1)

        x = torch.hstack((head_entity, relation, tail_entity))
        if self.apply_semantic_constraint:
            raise NotImplementedError()
        else:
            with torch.no_grad():
                out = self.model(x)
                if logits:
                    return out
                else:
                    return torch.sigmoid(out)

    def predict_conjunctive_query(self, entity: str, relations: List[str], topk: int = 3,
                                  show_intermediate_results=False) -> Set[str]:
        """
         Find an answer set for a conjunctive query.

         A graphical explanation is shown below
                                                    -> result_1
                                -> e_i,relations[1] -> result_2
                                                    -> result_3

                                                    -> result_4
         entity, relations[0]   -> e_j,relations[1] -> result_5
                                                    -> result_7

                                                    -> result_8
                                -> e_k,relations[1] -> result_9
                                                    -> result_10

        Parameter
        ---------
        entity: str

        String representation of a selected/anchor entity.

        relations: List[str]

        String representations of selected relations.

        topk: int

        Highest ranked k item.

        Returns: Tuple
        ---------

        Highest K scores and entities
        """

        assert isinstance(entity, str)
        assert isinstance(relations, list)
        assert len(entity) >= 1
        assert len(relations) >= 1
        # (1) An entity set.
        results = set()
        # (2) Bookkeeping.
        each_intermediate_result = dict()
        hop_counter = 0
        # (3) Iterate over each relation.
        for r in relations:
            # (3.1) if entity is an anchor entity:
            if len(results) == 0:
                top_ranked_entities = self.predict_topk(head_entity=[entity], relation=[r], topk=topk)[1]
                results = set(top_ranked_entities)
                each_intermediate_result[(hop_counter, entity, r)] = top_ranked_entities
            else:
                # (3.2) Iterative over intermediate results
                temp_intermediate_results = set()
                while results:
                    entity = results.pop()
                    top_ranked_entities = self.predict_topk(head_entity=[entity], relation=[r], topk=topk)[1]
                    temp_intermediate_results |= set(top_ranked_entities)
                    each_intermediate_result[(hop_counter, entity, r)] = top_ranked_entities
                # (3.3)
                results = temp_intermediate_results
            hop_counter += 1
        if show_intermediate_results is True:
            return results, each_intermediate_result
        else:
            return results

    def find_missing_triples(self, confidence: float, entities: List[str] = None, relations: List[str] = None,
                             topk: int = 10,
                             at_most: int = sys.maxsize) -> Set:
        """
         Find missing triples

         Iterative over a set of entities E and a set of relation R : \forall e \in E and \forall r \in R f(e,r,x)
         Return (e,r,x)\not\in G and  f(e,r,x) > confidence

        Parameter
        ---------
        confidence: float

        A threshold for an output of a sigmoid function given a triple.

        topk: int

        Highest ranked k item to select triples with f(e,r,x) > confidence .

        at_most: int

        Stop after finding at_most missing triples

        Returns: Set
        ---------

        {(e,r,x) | f(e,r,x) > confidence \land (e,r,x) \not\in G
        """

        assert 1.0 >= confidence >= 0.0
        assert topk >= 1

        def select(items: List[str], item_mapping: Dict[str, int]) -> Iterable[Tuple[str, int]]:
            """
             Get selected entities and their indexes

            Parameter
            ---------
            items: list

            item_mapping: dict


            Returns: Iterable
            ---------

            """

            if items is None:
                return item_mapping.items()
            else:
                return ((i, item_mapping[i]) for i in items)

        extended_triples = set()
        print(f'Number of entities:{len(self.entity_to_idx)} \t Number of relations:{len(self.relation_to_idx)}')

        # (5) Cartesian Product over entities and relations
        # (5.1) Iterate over entities
        print('Finding missing triples..')
        for str_head_entity, idx_entity in select(entities, self.entity_to_idx):
            # (5.1) Iterate over relations
            for str_relation, idx_relation in select(relations, self.relation_to_idx):
                # (5.2) \forall e \in Entities store a tuple of scoring_func(head,relation,e) and e
                # (5.3.) Sort (5.2) and return top  tuples
                predicted_scores, str_tail_entities = self.predict_topk(head_entity=[str_head_entity],
                                                                        relation=[str_relation], topk=topk)
                # (5.4) Iterate over 5.3
                for predicted_score, str_entity in zip(predicted_scores, str_tail_entities):
                    # (5.5) If score is less than 99% ignore it
                    if predicted_score < confidence:
                        break
                    else:
                        # /5.6) False if 0, otherwise 1
                        is_in = np.any(
                            np.all(self.train_set == [idx_entity, idx_relation, self.entity_to_idx[str_entity]],
                                   axis=1))
                        # (5.7) If (5.6) is true, ignore it
                        if is_in:
                            continue
                        else:
                            # (5.8) Remember it
                            extended_triples.add((str_head_entity, str_relation, str_entity))
                            print(f'Number of found missing triples: {len(extended_triples)}')
                            if len(extended_triples) == at_most:
                                return extended_triples
        return extended_triples

    def deploy(self, share: bool = False, top_k: int = 10):
        import gradio as gr

        def predict(str_subject: str, str_predicate: str, str_object: str, random_examples: bool):
            if random_examples:
                return random_prediction(self)
            else:
                if self.is_seen(entity=str_subject) and self.is_seen(
                        relation=str_predicate) and self.is_seen(entity=str_object):
                    """ Triple Prediction """
                    return deploy_triple_prediction(self, str_subject, str_predicate, str_object)

                elif self.is_seen(entity=str_subject) and self.is_seen(
                        relation=str_predicate):
                    """ Tail Entity Prediction """
                    return deploy_tail_entity_prediction(self, str_subject, str_predicate, top_k)
                elif self.is_seen(entity=str_object) and self.is_seen(
                        relation=str_predicate):
                    """ Head Entity Prediction """
                    return deploy_head_entity_prediction(self, str_object, str_predicate, top_k)
                elif self.is_seen(entity=str_subject) and self.is_seen(entity=str_object):
                    """ Relation Prediction """
                    return deploy_relation_prediction(self, str_subject, str_object, top_k)
                else:
                    KeyError('Uncovered scenario')
            # If user simply select submit
            return random_prediction(pre_trained_kge)

        gr.Interface(
            fn=predict,
            inputs=[gr.inputs.Textbox(lines=1, placeholder=None, label='Subject'),
                    gr.inputs.Textbox(lines=1, placeholder=None, label='Predicate'),
                    gr.inputs.Textbox(lines=1, placeholder=None, label='Object'), "checkbox"],
            outputs=[gr.outputs.Textbox(label='Input Triple'),
                     gr.outputs.Dataframe(label='Outputs', type='pandas')],
            title=f'{self.name} Deployment',
            description='1. Enter a triple to compute its score,\n'
                        '2. Enter a subject and predicate pair to obtain most likely top ten entities or\n'
                        '3. Checked the random examples box and click submit').launch(share=share)

    # @TODO: Do we really need this ?!
    def train_triples(self, head_entity: List[str], relation: List[str], tail_entity: List[str], labels: List[float],
                      iteration=2, optimizer=None):
        """

        :param head_entity:
        :param relation:
        :param tail_entity:
        :param labels:
        :param iteration:
        :param lr:
        :return:
        """
        assert len(head_entity) == len(relation) == len(tail_entity) == len(labels)
        # (1) From List of strings to TorchLongTensor.
        x = torch.LongTensor(self.index_triple(head_entity, relation, tail_entity)).reshape(1, 3)
        # (2) From List of float to Torch Tensor.
        labels = torch.FloatTensor(labels)
        # (3) Train mode.
        self.set_model_train_mode()
        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=0.1)
        print(f'Iteration starts...')
        # (4) Train.
        for epoch in range(iteration):
            optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.model.loss(outputs, labels)
            print(f"Iteration:{epoch}\t Loss:{loss.item()}\t Outputs:{outputs.detach().mean()}")
            loss.backward()
            optimizer.step()
        # (5) Eval
        self.set_model_eval_mode()
        with torch.no_grad():
            outputs = self.model(x)
            loss = self.model.loss(outputs, labels)
            print(f"Eval Mode:\tLoss:{loss.item()}")

    def train_k_vs_all(self, head_entity, relation, iteration=1, lr=.001):
        """
        Train k vs all
        :param head_entity:
        :param relation:
        :param iteration:
        :param lr:
        :return:
        """
        assert len(head_entity) == 1
        # (1) Construct input and output
        out = self.construct_input_and_output_k_vs_all(head_entity, relation)
        if out is None:
            return
        x, labels, idx_tails = out
        # (2) Train mode
        self.set_model_train_mode()
        # (3) Initialize optimizer # SGD considerably faster than ADAM.
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=.00001)

        print('\nIteration starts.')
        # (3) Iterative training.
        for epoch in range(iteration):
            optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.model.loss(outputs, labels)
            if len(idx_tails) > 0:
                print(
                    f"Iteration:{epoch}\t Loss:{loss.item()}\t Avg. Logits for correct tails: {outputs[0, idx_tails].flatten().mean().detach()}")
            else:
                print(
                    f"Iteration:{epoch}\t Loss:{loss.item()}\t Avg. Logits for all negatives: {outputs[0].flatten().mean().detach()}")

            loss.backward()
            optimizer.step()
            if loss.item() < .00001:
                print(f'loss is {loss.item():.3f}. Converged !!!')
                break
        # (4) Eval mode
        self.set_model_eval_mode()
        with torch.no_grad():
            outputs = self.model(x)
            loss = self.model.loss(outputs, labels)
        print(f"Eval Mode:Loss:{loss.item():.4f}\t Outputs:{outputs[0, idx_tails].flatten().detach()}\n")

    def train(self, kg, lr=.1, epoch=10, batch_size=32, neg_sample_ratio=10, num_workers=1) -> None:
        """ Retrained a pretrain model on an input KG via negative sampling."""
        # (1) Create Negative Sampling Setting for training
        print('Creating Dataset...')
        train_set = TriplePredictionDataset(kg.train_set,
                                            num_entities=len(kg.entity_to_idx),
                                            num_relations=len(kg.relation_to_idx),
                                            neg_sample_ratio=neg_sample_ratio)
        num_data_point = len(train_set)
        print('Number of data points: ', num_data_point)
        train_dataloader = DataLoader(train_set, batch_size=batch_size,
                                      #  shuffle => to have the data reshuffled at every epoc
                                      shuffle=True, num_workers=num_workers,
                                      collate_fn=train_set.collate_fn, pin_memory=True)

        # (2) Go through valid triples + corrupted triples and compute scores.
        # Average loss per triple is stored. This will be used  to indicate whether we learned something.
        print('First Eval..')
        self.set_model_eval_mode()
        first_avg_loss_per_triple = 0
        for x, y in train_dataloader:
            pred = self.model(x)
            first_avg_loss_per_triple += self.model.loss(pred, y)
        first_avg_loss_per_triple /= num_data_point
        print(first_avg_loss_per_triple)
        # (3) Prepare Model for Training
        self.set_model_train_mode()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        print('Training Starts...')
        for epoch in range(epoch):  # loop over the dataset multiple times
            epoch_loss = 0
            for x, y in train_dataloader:
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(x)
                loss = self.model.loss(outputs, y)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
            print(f'Epoch={epoch}\t Avg. Loss per epoch: {epoch_loss / num_data_point:.3f}')
        # (5) Prepare For Saving
        self.set_model_eval_mode()
        print('Eval starts...')
        # (6) Eval model on training data to check how much an Improvement
        last_avg_loss_per_triple = 0
        for x, y in train_dataloader:
            pred = self.model(x)
            last_avg_loss_per_triple += self.model.loss(pred, y)
        last_avg_loss_per_triple /= len(train_set)
        print(f'On average Improvement: {first_avg_loss_per_triple - last_avg_loss_per_triple:.3f}')
