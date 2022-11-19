from utils.nlg_utils import read_json_data

def generate_description(data, apred, predicate_nlg_key='description'):
    """
    :param data: json data list with list of all predicates
    :param apred: the predicate to be translated to natural language
    :return: human readable language
    """
    apred_name = apred[0]
    for pred in data:
        if (apred_name == pred['name']):
            apred_numargs = pred['numargs']
            apred_descp = pred[predicate_nlg_key]
            text = apred_descp
            for a in range(apred_numargs):
                text = text.replace('$' + str(a + 1), apred[a + 1])
            break
    return text

def pred_description_generation(predicate_list):
    """
    :param predicate_list: predicate list for which we genrate description
    :return: generated description list of predicate
    """
    preds_list = read_json_data()

    generate_description_list = []
    nlg_predicate = []
    count = 0
    for predicate in predicate_list:
        predicate_name = predicate.split("(")[0]
        argument_in_predicate =  predicate[predicate.find('(') + 1:predicate.find(')')].split(',')
        nlg_predicate.append(generate_description(preds_list, [predicate_name, *argument_in_predicate]))
        count = count + 1
    generate_description_list.append(" ".join(nlg_predicate))
    return generate_description_list
