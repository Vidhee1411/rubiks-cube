import os
import re
import json

def get_precondition(learned_macro_action):
    """
    :param learned_macro_action: learned macro action
    :return: precondition clauses
    """
    precondition_list = learned_macro_action.get_precond().clauses
    precondition_list_clauses = [str(clause) for clause in precondition_list]
    return precondition_list_clauses

def preprocess_precondition(precondition_learned_list):
    """
    :param precondition_learned_list: learned precondition in list format
    :return: predicate list
    """
    for precondition_learned in precondition_learned_list:
        predicate_list = precondition_learned.split(":-")
        regx_for_comma_separation = re.compile(r",(?![^(]*\))")
        separate_predicate = regx_for_comma_separation.split(predicate_list[1].replace(" ","").replace(".", "").replace("-", "").replace("+", ""))
    return separate_predicate

def read_json_data(root_dir_path='nlg_pred_gen', data_dir_name='data', data_file_name='template.json'):
    """
    :param data_dir_name: name of data directory
    :param data_file_name: name of data_file_path we have to retrieve json data from
    :return: json data list predicate list
    """
    data_dir_path = os.path.join(root_dir_path, data_dir_name)
    data_file_path = os.path.join(data_dir_path, data_file_name)

    # Opening JSON file
    json_file_object = open(data_file_path)

    # returns JSON object as
    # a dictionary
    preds = json.load(json_file_object)

    # Closing file
    json_file_object.close()

    preds_list = preds['predicates']
    return preds_list