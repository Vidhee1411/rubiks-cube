from typing import Dict, List
from pyswip import Prolog


def main():
    prolog = Prolog()
    colors = ["w", "o", "r", "g", "b", "y"]

    # center cubelets
    prolog.assertz("cubelet(X):-center(X)")
    for color in colors:
        prolog.assertz(f"center({color}_c)")

        sticker_name: str = f"{color}_{color}_c"
        prolog.assertz(f"sticker({sticker_name})")
        prolog.assertz(f"sticker_type({sticker_name},c)")

    # opposite centers
    prolog.assertz("opposite_face(X,Y) :- opposite_face_sym(X,Y),sticker(X),sticker(Y)")
    prolog.assertz("opposite_face(X,Y) :- opposite_face_sym(Y,X),sticker(X),sticker(Y)")
    prolog.assertz("opposite_face_sym(w_w_c,y_y_c)")
    prolog.assertz("opposite_face_sym(r_r_c,o_o_c)")
    prolog.assertz("opposite_face_sym(b_b_c,g_g_c)")

    # clockwise
    clockwise: Dict[str, List[str]] = dict()
    clockwise["w_w_c"] = ["b_b_c", "r_r_c", "g_g_c", "o_o_c"]
    clockwise["r_r_c"] = ["y_y_c", "g_g_c", "w_w_c", "b_b_c"]
    clockwise["b_b_c"] = ["r_r_c", "w_w_c", "o_o_c", "y_y_c"]

    for center in clockwise.keys():
        for idx in range(len(clockwise[center])):
            c1 = clockwise[center][idx]
            c2 = clockwise[center][(idx + 1) % len(clockwise[center])]
            prolog.assertz(f"clockwise({center},{c1},{c2})")

            center_opps = list(prolog.query(f"opposite_face({center},X)"))
            assert len(center_opps) == 1
            prolog.assertz(f"clockwise({center_opps[0]['X']},{c2},{c1})")

    # counter clockwise
    prolog.assertz("counterclockwise(X,Y,Z):-clockwise(X,Z,Y),center(X),center(Y),center(Z)")

    # adjacent centers
    prolog.assertz("adjacent(X,Y):-clockwise(X,Y,Z)")

    # edge cubelets
    prolog.assertz("cubelet(X):-edge(X)")
    for center_adj in list(prolog.query("adjacent(w_w_c,X)")):
        prolog.assertz(f"edge(w{center_adj['X'][0]}_e)")
    for center_adj in list(prolog.query("adjacent(y_y_c,X)")):
        prolog.assertz(f"edge(y{center_adj['X'][0]}_e)")
    prolog.assertz("edge(br_e)")
    prolog.assertz("edge(bo_e)")
    prolog.assertz("edge(go_e)")
    prolog.assertz("edge(gr_e)")

    prolog.assertz("same_face(X,Y) :- same_face_sym(X,Y); same_face_sym(Y,X)")
    prolog.assertz("not(same_face(X,Y)) :- not(same_face_sym(X,Y)); not(same_face_sym(Y,X))")
    for edge_q in list(prolog.query("edge(X)")):
        edge = edge_q['X']
        sticker1_name: str = f"{edge[0]}_{edge}"
        sticker2_name: str = f"{edge[1]}_{edge}"

        prolog.assertz(f"sticker({sticker1_name})")
        prolog.assertz(f"sticker({sticker2_name})")

        prolog.assertz(f"sticker_type({sticker1_name},e)")
        prolog.assertz(f"sticker_type({sticker2_name},e)")

        prolog.assertz(f"not(same_face_sym({sticker2_name},Y)) :- same_face_sym({sticker1_name},X),opposite_face(X,Y)")
        prolog.assertz(f"not(same_face_sym({sticker1_name},Y)) :- same_face_sym({sticker2_name},X),opposite_face(X,Y)")

    prolog.assertz(f"not(same_face_sym(X,Z)) :- same_face_sym(X,Y),opposite_face(Y,Z)")
    prolog.assertz(f"not(same_face_sym(X,Z)) :- same_face_sym(X,Y),adjacent(Y,Z)")

    # solved
    prolog.assertz("is_solved :- same_face_sym(w_wo_e,w_w_c),same_face_sym(o_wo_e,o_o_c)")

    prolog.assertz("same_face_sym(w_wo_e,b_b_c)")
    prolog.assertz("same_face_sym(o_wo_e,o_o_c)")

    print(list(prolog.query("is_solved")))
    print(list(prolog.query("not(same_face(w_wo_e,Y))")))


if __name__ == "__main__":
    main()
