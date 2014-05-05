# coding: utf-8
from data_readers import get_matches, separate_samples, normalize
from sklearn.neighbors import KNeighborsClassifier
from random import random


def train(train_inputs, train_outputs, test_inputs, test_outputs,
          neighbors=10):
    classifier = KNeighborsClassifier(n_neighbors=neighbors)
    classifier.fit(train_inputs, train_outputs)

    score = classifier.score(test_inputs, test_outputs)

    return classifier, score


def full_train(input_features=None, output_feature='winner', remove_ties=False):
    if input_features is None:
        input_features = ['matches_won_percent',
                          'podium_score_yearly',
                          'matches_won_percent_2',
                          'podium_score_yearly_2']

    matches = get_matches(with_team_stats=True,
                          duplicate_with_reversed=True)

    if remove_ties:
        matches = matches[matches['score1'] != matches['score2']]

    inputs, outputs = separate_samples(matches,
                                       input_features,
                                       output_feature)

    inputs = normalize(inputs)

    train_inputs, train_outputs, test_inputs, test_outputs = split_samples(inputs, outputs)

    predictor, score = train(train_inputs,
                             train_outputs,
                             test_inputs,
                             test_outputs)

    return predictor, score
