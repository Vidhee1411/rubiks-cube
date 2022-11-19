# DeepXube
`python scripts/deepxube.py <results_directory>`
The output of the entire script gets written to the results directory.

The results for the Rubik's cube with symmetric actions has been obtained. One can run:
`python scripts/deepxube.py results/cube3_sym/`.

One can run the code from scratch, which will take over a couple hours on a CPU, by using the `--redo` switch, 
which will erase the current data, or by specifying a different directory.

## Macro actions and Preconditions
The function `get_m_acts` returns a list of macro actions.

For each macro action `m_act`, one can obtain the program for its precondition: `m_act.get_precond()`, 
as well as print it to the screen: `program_utils.print_program(m_act.get_precond())`.

## Debugging
To make debugging easier when doing `pdb.set_trace()`, use `--debug`.
This prevents the output from being written to the results file and makes deubbing on the command line easier.

## Decision tree
The function `induce_dt` has the code to produce the training data. This takes about 90 seconds.
Extra logic and arguments can be added to allow one to save the data to a file and load it to avoid computation time.

# Rubik's Cube
## Visualizer
`python visualizers/cube3_viz.py --subgoal_num <subgoal_num>`

## Subgoals
subgoal_num is which subgoal we are currently trying to achieve. 0 is the same as trying to solve the entire puzzle.
There is currently 1 and 2. You can visualize them with
`python visualizers/cube3_viz.py --subgoal_num 1`
`python visualizers/cube3_viz.py --subgoal_num 2` 

## Moving the cube

You can change the perspective by clicking with the mouse

You can move the cube with the U, D, L, R, B, F buttons on the keyboard. Hold tab to move the cube counter-clockwise.

# Implementing a New Environment
See `environments/environment_abstract.py` to see the functions that need to be implemented.

See `environments/cube3.py` for an example implementation.

The nlg is being geenrated inside function get_m_act() for deepxube.py.

The helper functions to process predicate is insde utils/nlg_utils.py.

The template to generate nlg is inside nlg_pred_gen/data/template.json.

The function to generate nlg is inside nlg_pred_gen/pred_description_gen.py.


# Demo Display

Code: scripts/demo_display <br />
Results used : cube3_sym <br />
Run the code: <br />
  `python scripts/demo_display` <br />
or  <br />
`from scripts import demo_display`<br />
`num = x # x is the number of examples you want to obtain`<br />
`states_colors, moves_tobeperformed, nlg_precond = demo_display.get_states_examples(num)`<br />
