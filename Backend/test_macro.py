import pickle
import re
import json
import os

def get_macro_test():
    save_file = "/Users/rojinapanta/PycharmProject/DeepCubeA_Teach/results/cube3_sym/progress.pkl"
    states_p, is_solved, m_acts_learned, m_acts_ban, itr =  pickle.load(open(save_file, "rb"))

    macro_action_moves_list = m_acts_learned[0].get_macro_action().clauses
    macro_list_clauses = [str(clause) for clause in macro_action_moves_list]
    # print(str(m_acts_learned[0].get_macro_action().clauses[0]))
    # print(macro_list_clauses)
    # print(len(macro_list_clauses))
    precondition_list = m_acts_learned[0].get_precond().clauses
    # print(precondition_list)
    precondition_list_clauses = [str(clause) for clause in precondition_list]
    # print(str(m_acts_learned[0].get_precond().clauses[0]))
    # print(precondition_list_clauses)

    # print(states_p)
    # print(len(m_acts_learned))
    # print(len(precondition_list_clauses))
    count = 0
    macro_Action_list1 = []
    precondition_list_clauses_list = []
    for macro_action_learned in m_acts_learned:
        macro_action_moves_list = macro_action_learned.get_macro_action().clauses
        # import pdb
        # pdb.set_trace()
        macro_list_clauses = [str(clause) for clause in macro_action_moves_list]
        # print(str(m_acts_learned[0].get_macro_action().clauses[0]))
        # print(macro_list_clauses)
        # print(len(macro_list_clauses))
        # import pdb
        # pdb.set_trace()
        precondition_list = macro_action_learned.get_precond().clauses
        # print(precondition_list)
        precondition_list_clauses = [str(clause) for clause in precondition_list]
        precondition_list_clauses_list.append(precondition_list_clauses)
        # print(str(m_acts_learned[0].get_precond().clauses[0]))
        # print(precondition_list_clauses)
        macro_Action_list1.append(macro_list_clauses)
        # print(states_p)
        # print(len(m_acts_learned))
        # print(len(precondition_list_clauses))
        count+=1
        # print(count)
    return precondition_list_clauses_list, macro_Action_list1

def pred_description_generation():
    predicate_list, macro_Action_move_list = preprocess_precondition_list()

    preds_list = read_json_data()

    generate_description_list = []
    count = 0
    for predicates in predicate_list:
        # print(predicates)
        nlg_predicate = []
        for predicate in predicates:
            # print(predicate)
            predicate_name = predicate.split("(")[0]
            argument_in_predicate =  predicate[predicate.find('(') + 1:predicate.find(')')].split(',')
            # print(argument_in_predicate)
            # exit(0)
            nlg_predicate.append(generate_description(preds_list, [predicate_name, *argument_in_predicate]))
            # print(nlg_predicate)
            # print(argument_in_predicate)
        # print(macro_Action_move_list[count])
        print(predicates)
        print(" ".join(nlg_predicate))
        count = count + 1
        generate_description_list.append(" ".join(nlg_predicate))
    # print(len(generate_description_list))

def read_json_data(root_dir_path="/Users/rojinapanta/PycharmProject/DeepCubeA_Teach/nlg_pred_gen", data_dir_name='data', data_file_name='template.json'):
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

    # Iterating through the json list
    # for i in preds['predicates']:
    #     print(i)

    preds_list = preds['predicates']
    return preds_list

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
                # import pdb
                # pdb.set_trace()
                text = text.replace('$' + str(a + 1), apred[a + 1])
            break
    return text

def preprocess_precondition_list():
    precondition_list, macro_action_move_list = get_macro_test()
    # import pdb
    # pdb.set_trace()
    separate_predicate_list = []
    for pre_list in precondition_list:
        predicate_list = pre_list[0].split(":-")
        # print(predicate_list[1])
        regx_for_comma_separation = re.compile(r",(?![^(]*\))")
        separate_predicate = regx_for_comma_separation.split(predicate_list[1].replace(" ","").replace(".", "").replace("-", "").replace("+", ""))
        # print(separate_predicate)
        separate_predicate_list.append(separate_predicate)
        # print(separate_predicate[0])
    return separate_predicate_list, macro_action_move_list

pred_description_generation()