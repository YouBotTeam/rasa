from collections import Counter
from typing import Dict, Text, Any, List
import numpy as np

AVG = 'avg'
SUM = 'sum'
DOT = 'dot'
CONFIDENCE_OP = [AVG, SUM, DOT]
UNCLASSIFIED_INTENT = 'unclassified'


def intent_maj_voting(intent_result: Dict[Text, Any]):
    """

    :param
        intent_result: dict{
                                'value': dict{ 'name':         str
                                               'confidence':   float}

                                'ranking':      dict{ list[dict{  'value':        str,
                                                                  'confidence':   float},
                                                           dict{ ... },
                                                             .
                                                             .
                                                             .]

                                'extractor':    str
                              }
    :return:
        maj_voting: dict {'name': intent_name,
                          'confidence': confidence,
                          'model_intents': models_intent,
                          'strategy': CONFIDENCE_OP}

        intent_idx: index of the chosen model
    """

    # models_intent: list[ models ]
    #
    #     models: dict{ 'name':         str     (intent_name),
    #                   'confidence':   float   (intent_confidence),
    #                   'extractor':    str     (model_name),
    #                   'model_n':      int     (model_index list)}

    models_intent = []

    for key in intent_result:
        item = {}
        item['name'] = intent_result[key]['value']['name']
        item['id'] = intent_result[key]['value']['id']
        item['confidence'] = intent_result[key]['value']['confidence']

        item['extractor'] = intent_result[key]['extractor']
        item['model_n'] = intent_result[key]['model_n']
        models_intent.append(item)

    value_counter = Counter([intent['name'] for intent in models_intent])

    most_commons = value_counter.most_common()
    second_common = None
    # check_if_majority --> check if the list contains a majority for one value

    if len(most_commons)>1:
        first_common, second_common = most_commons[0], most_commons[1]
        check_if_majority = first_common[1] != second_common[1]
    else:
        check_if_majority = True

    if check_if_majority:
        intent_name = value_counter.most_common(1)[0][0]
        second_intent = second_common[0] if second_common is not None else 'None'
        confidences = compute_confidence(models_intent, intent_name)

        max_confidence = -1
        intent_idx = -1

        for intent in models_intent:
            if (intent['name'] == intent_name and
                    intent['confidence'] > max_confidence):

                max_confidence = intent['confidence']
                intent_idx = intent['model_n']

    else:
        first_intent, second_intent = first_common[0], second_common[0]
        if first_common[1] == 1 and second_common[1] == 1:
            confidence_list = [intent['value']['confidence'] for intent in intent_result.values()]
            intent_idx = confidence_list.index(max(confidence_list))
        else:
            first_confidence, first_intent_idx = get_avg_confidence_from_intent(intent_result, first_intent)
            second_confidence, second_intent_idx = get_avg_confidence_from_intent(intent_result, second_intent)

            if first_confidence > second_confidence:
                intent_idx = first_intent_idx
            else:
                intent_idx = second_intent_idx

        # intent_name = models_intent[intent_idx]['name']
        # confidences = {conf_op: models_intent[intent_idx]['confidence'] for conf_op in CONFIDENCE_OP}

        # Provide the OTHER intent
        intent_name = UNCLASSIFIED_INTENT
        confidences = {conf_op: 1-models_intent[intent_idx]['confidence'] for conf_op in CONFIDENCE_OP}

    # maj_intent = {'name': intent_name,
    #               'second_intent': second_intent,
    #               'confidence': confidences[AVG],
    #               'avg_confidence': confidences[AVG],
    #               'sum_confidence': confidences[SUM],
    #               'dot_confidence': confidences[DOT],
    #               'majority': check_if_majority,
    #               'models_intent': models_intent}

    maj_intent = {'name': intent_name,
                  'id': hash(intent_name),
                  'confidence': confidences[AVG],

                  'second_intent': second_intent,
                  'majority': check_if_majority,
                  }

    intent_ranking = [item for item in models_intent]

    return maj_intent, intent_ranking, intent_idx


def get_model_from_intent(intent_result, intent):

    models = []

    for model in intent_result.values():
        if model['value']['name'] == intent:
            models.append(model)

    return models


def get_avg_confidence_from_intent(intent_result, intent):

    models = get_model_from_intent(intent_result, intent)
    confidences = []
    max_confidence_idx = 0

    for model in models:
        confidence = model['value']['confidence']
        confidences.append(confidence)
        if confidence > max_confidence_idx:
            max_confidence_idx = confidence
            model_idx = model['model_n']

    return np.mean(confidences), model_idx


def compute_confidence(models_intent: List[Dict[Text, Any]],
                       intent_name: Text):
    confidences = {}

    for conf_op in CONFIDENCE_OP:
        if (conf_op == AVG):
            operation = np.mean
        elif (conf_op == SUM):
            operation = np.sum
        elif (conf_op == DOT):
            operation = np.prod
        else:
            operation = None

        confidence = operation([intent['confidence'] for intent in models_intent
                                if intent['name'] == intent_name])

        confidences[conf_op] = confidence

    return confidences
