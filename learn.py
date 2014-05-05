# coding: utf-8
from data_readers import get_matches, get_team_stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from random import random
from matplotlib import pyplot
import pygal


def get_samples(origin_features=None, result_feature='winner'):
    if not origin_features:
        origin_features = (
            'matches_won_percent',
            'cups_won_yearly',
            'podium_score_yearly',
            'matches_won_percent_2',
            'cups_won_yearly_2',
            'podium_score_yearly_2',
        )

    matches = get_matches(duplicate_with_reversed=True)
    stats = get_team_stats()

    # add teams stats to the matches
    matches = matches.join(stats, on='team1')\
                     .join(stats, on='team2', rsuffix='_2')

    inputs = [tuple(matches.loc[i, feature]
                    for feature in origin_features)
              for i in matches.index]

    outputs = tuple(matches[result_feature].values)

    assert len(inputs) == len(outputs)

    return inputs, outputs


def graph_samples(inputs, outputs, feature_x_index, feature_y_index, graph_path='samples.svg'):
    groups = {}
    for i, inputs_row in enumerate(inputs):
        output = outputs[i]
        if output not in groups:
            groups[output] = []
        groups[output].append((inputs_row[feature_x_index], inputs_row[feature_y_index]))

    chart = pygal.XY(stroke=False,
                     title='Samples',
                     style=pygal.style.CleanStyle)

    for group_name, points in groups.items():
        chart.add(str(group_name), points)

    chart.render_to_file(graph_path)


def train(train_inputs, train_outputs, test_inputs, test_outputs,
          neighbors=10):
    classifier = KNeighborsClassifier(n_neighbors=neighbors)
    classifier.fit(train_inputs, train_outputs)

    score = classifier.score(test_inputs, test_outputs)

    return classifier, score


def normalize(inputs):
    scaler = StandardScaler()
    new_inputs = scaler.fit_transform(inputs)

    return new_inputs


def full_train():
    inputs, outputs = get_samples()
    inputs = normalize(inputs)

    train_inputs, train_outputs, test_inputs, test_outputs = split_samples(inputs, outputs)

    predictor, score = train(train_inputs, train_outputs, test_inputs, test_outputs)

    return predictor, score


def split_samples(inputs, outputs, percent=0.75):
    assert len(inputs) == len(outputs)

    inputs1 = []
    inputs2 = []
    outputs1 = []
    outputs2 = []

    for i, inputs_row in enumerate(inputs):
        if random() < percent:
            input_to = inputs1
            output_to = outputs1
        else:
            input_to = inputs2
            output_to = outputs2

        input_to.append(inputs_row)
        output_to.append(outputs[i])

    return inputs1, outputs1, inputs2, outputs2
