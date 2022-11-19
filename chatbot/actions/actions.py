from asyncio.windows_events import NULL
from typing import Any, Text, Dict, List
import rasa.core.tracker_store
from rasa.shared.core.trackers import DialogueStateTracker
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import json
# from . import server
from firebase import firebase
import numpy as np
import sys
import pickle5 as pkl
import random
import time
import random

from sqlalchemy import false, true

firebase = firebase.FirebaseApplication(
    "https://allure-chatbot-default-rtdb.firebaseio.com/", None)
with open('nlg_for_progress_file.pkl', 'rb') as f:
    nlg = pkl.load(f)

with open('display_data.pkl', 'rb') as f:
    var = pkl.load(f)

with open('explanation.pkl', 'rb') as f:
    explanation = pkl.load(f)
# solve = false

user = 'user1'
firebase.put('/', user+'/solve', "false")
firebase.put('/', user+'/state', "null")
firebase.put('/', user+'/moves', "null")


class init(Action):
    def name(self) -> Text:
        return "action_init"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        solve = false
        # firebase.put('/', "solve", "False")
        message = tracker.latest_message["text"]
        print("Message: " + message)
        macro_action = int(tracker.latest_message['entities'][0]['value'])
        if(macro_action < 1 or macro_action > 12):
            macro_action = 1
        print("Macro-Action:", macro_action)
        i = macro_action

        # Length of number of samples contained in one macro action example
        lth = len(explanation[i][2])

        # First index[this]:     1-9, no zero
        # Second index[][this]:  2 Contains list of states
        #                       3 Contains list of moves
        # [1-12][2][n]:          Contains one set of state(54 Length Array)
        # [1-12][3][n]:          Contains one set of move

        lst = list(range(0, lth))
        r = random.choice(lst)
        state = np.array(explanation[i][2][r]).tolist()
        moves = explanation[i][3][r]
        moves = ''.join(moves)
        moves = moves.lower()
        print("State:", state, "\nMoves:", moves,
              "\n0or-1:", explanation[i][1])

        print("\n", explanation[i], "\n")
        #print("Type", type(moves))
        print("Random: ", r)
        data = {'solve': 'false', 'state': state, 'moves': moves}
        firebase.put('/', user, data)
        # print(nlgText)
        #nlgText = nlgText.replace(". ", "\n\n")
        next_action_text = ""
        next_action = explanation[i][1]
        # if explanation[i][1]!=0:
        #     next_action_text = "This action is related to macro action "+str(next_action)
        # dispatcher.utter_message(text="Performing macro action: "+str(i) + "\n\nProlog: " + explanation[i][0] + "\n\n"+next_action_text)
        # dispatcher.utter_message(text="Moves performed: " + str(moves))

        explanation_string = explanation[i][0].strip()
        explanation_array = explanation_string.split('. ')
        print(explanation_array)
        if explanation[i][1] != 0:
            next_action_text = "After you have performed these moves, you can use the skills you learned in Scenario " + \
                str(next_action)+" to put the white-green edge piece in place."
        dispatcher.utter_message(text="Scenario: "+str(i))

        # for x in range(len(explanation_array)):
        #     dispatcher.utter_message(text= "\n\n" + x)

        for ex in explanation_array:
            if(ex != ''):
                if(":" in ex):
                    moves_for_scenario = explanation[i][3][r]
                    moves_for_scenario = '\t'.join(moves)
                    dispatcher.utter_message(
                        text="\n\n" + ex + " " + moves_for_scenario)
                else:
                    dispatcher.utter_message(text="\n\n" + ex)

        if(explanation[i][1] != 0):
            dispatcher.utter_message(text="\n\n" + next_action_text)

        dispatcher.utter_message(text="Click solve whenever you are ready", buttons=[
                                 {"title": 'Solve', "payload": "solve"}])
        return []


class solveMoves(Action):
    def name(self) -> Text:
        return "action_solve"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        solve = true
        firebase.put('/', user+'/solve/', "true")
        time.sleep(0.1)
        firebase.put('/', user+'/solve', "false")
        firebase.put('/', user+'/state', "null")
        firebase.put('/', user+'/moves', "null")
        dispatcher.utter_message(
            text="Watch the cube to see the moves being performed.")


#White Cross user-task changes
class userTask(Action):
    def name(self) -> Text:
        return "action_user_task"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain : Dict[Text, Any]) -> List[Dict[Text, Any]]:
          #initial_configuration
          initial_state = []
          initial_state.append([1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 18, 18, 18, 18, 18, 18, 18, 18, 18, 27, 27, 27, 27, 27, 27, 27, 27, 27, 36, 36, 36, 36, 36, 36, 36,
                               36, 36, 45, 45, 45, 45, 45, 45, 45, 45, 45])
          initial_state.append([27, 9, 18, 1, 1, 1, 27, 1, 18, 9, 27, 18, 45, 9, 18, 27, 1, 9, 18, 36, 36, 9, 18, 45, 1, 18, 1, 36, 27, 1, 18, 27, 27, 9, 18, 9, 45, 45, 45, 27, 36, 36, 36,
                               9, 1, 45, 9, 45, 36, 45, 45, 27, 36, 36])
          initial_state.append([36, 1, 27, 36, 1, 18, 45, 45, 1, 9, 18, 1, 45, 9, 9, 9, 1, 45, 45, 1, 1, 36, 18, 36, 27, 9, 27, 1, 27, 27, 27, 27, 1, 36, 9, 36, 18, 45, 18, 18, 36, 9, 18,
                               18, 36, 45, 36, 9, 27, 45, 27, 18, 45, 9])

          #Insert initial_configuration into firebase
          firebase.put('/', user+'/solve', "false")
          firebase.put('/', user+'/state', initial_state[random.randint(1,2)])
          firebase.put('/', user+'/moves', "null")

          cube_solved = False
          while not cube_solved:
              if firebase.get('/', user+'/solve/') == "true":
                cube_solved = True
              
              #print(tracker.latest_message)

          else:
            #Once we reach goal state
            dispatcher.utter_message(text = "Cube solved!")
            dispatcher.utter_message(response = "utter_crossdone")

    