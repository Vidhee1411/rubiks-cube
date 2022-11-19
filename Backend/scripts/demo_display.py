import deepxube
from environments.environment_abstract import Environment, State, MacroAction
from environments.cube3 import Cube3
from utils.prolog_utils import PrologProc, using_prog
import pickle
from utils import nnet_utils, program_utils, data_utils, viz_utils
from nlg_pred_gen.pred_description_gen import pred_description_generation
from utils.nlg_utils import get_precondition, preprocess_precondition
import numpy as np

def get_moves(m_act_learned):
    col_dict = {"r":"r", "o":"l", "y":"d", "b":"b", "w":"u", "g":"f"}
    dir_dict = {"cl":"","cc":"'"}
    temp = m_act_learned
    print(temp)
    temp = temp.split(" ")[1]
    temp = temp.split("move")
    temp = [i for i in temp if len(i)>2]
    temp = [i.split(",") for i in temp]
    temp = [(i[1],i[2]) for i in temp]
    temp = [col_dict[i[0]]+dir_dict[i[1]] for i in temp]

    return "".join(temp)

def get_states_examples(num):
    count = 0
    env: Environment = Cube3(1)
    env.generate_bk("results/cube3_sym/bk.pl")
    prolog: PrologProc = PrologProc()
    prolog.consult("results/cube3_sym/bk.pl")
    salient_start: str = "salient_start(A) :- true"
    salient_transition: str = "salient_t(A,B) :- not_in_place(A,Cbl),in_place(B,Cbl),in_place_subset(A,B), !"
    prolog.assertz(salient_transition)
    prolog.assertz(salient_start)
    device, devices, on_gpu = nnet_utils.get_device()
    heuristic_fn = nnet_utils.load_heuristic_fn("saved_models/cube3m/current/", device, on_gpu, env.get_nnet(),
                                                env, clip_zero=True, batch_size=10000)
    states_p, is_solved, m_acts_learned, m_acts_ban, itr = pickle.load(open("results/cube3_sym/progress.pkl", "rb"))
    m_acts_learned = deepxube.get_m_acts(env,"results/cube3_sym" , states_p, is_solved, m_acts_learned,
                                                   m_acts_ban, prolog, heuristic_fn, itr, "results/cube3_sym/progress.pkl",
                                                   0, False, False)
    state_unsolved = deepxube.generate_states_p(env, 100)
    states_check, states_moved, move_data = deepxube.apply_macro_actions(state_unsolved, prolog, m_acts_learned)
    is_solved_moved = [len(prolog.query(f"solved({x})")) > 0 for x in state_unsolved]
    generate = list()
    state_color = list()
    move_perform = list()
    for s, flag, flag2, m in zip(states_check, states_moved,is_solved_moved, move_data):
        if(flag):
            if(flag2):
                print("Yes")
                top_gen = list()
                mov_list = list()
                state_color.append(env.predicate_to_state([m[0][0]])[0].colors)
                for mov in m:
                    mov_list.append(get_moves(m_acts_learned[mov[1]].to_string()))
                    precondition_list = get_precondition(m_acts_learned[mov[1]])
                    predicate_list = preprocess_precondition(precondition_list)
                    generated_description = pred_description_generation(predicate_list)
                    top_gen.append(generated_description[0])
                move_perform.append(mov_list)
                generate.append(top_gen)
                count = count +1
                if count == num:
                    break 
    return state_color, move_perform, generate

def main():

    print(get_states_examples(1))
    
    return None

if __name__ == '__main__':
    main()