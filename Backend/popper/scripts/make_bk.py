from typing import List, Dict


def main():
    file = open("popper/examples/cube/bk.pl", "w")
    stickers: List[str] = ['w', 'y', 'o', 'b', 'g', 'r']
    clockwise: Dict[str, List[str]] = dict()
    clockwise["w"] = ["b", "r", "g", "o"]
    clockwise["o"] = ["y", "b", "w", "g"]
    clockwise["g"] = ["o", "w", "r", "y"]

    opposite: Dict[str, str] = {"w": "y", "o": "r", "g": "b"}

    file.write("%% stickers and cubelets\n")
    for sticker in stickers:
        file.write(f"sticker({sticker}).\n")
    file.write("\n")

    # file.write("cubelet(X) :- center(X).\n")
    # file.write("cubelet(X) :- edge(X).\n")
    # file.write("\n")

    # file.write(f"edge(wg_e).\n")
    # file.write("\n")

    # file.write("edge_has_sticker(wo_e, w).\n")
    # file.write("edge_has_sticker(wo_e, o).\n")
    # file.write("\n")

    file.write("%% Directions\n")
    file.write("direction(cl).\n")
    file.write("direction(cc).\n")
    file.write("\n")

    file.write("direction_opposite(cc, cl).\n")
    file.write("direction_opposite(cl, cc).\n")
    file.write("\n")

    for center, center_opp in opposite.items():
        file.write(f"face_opposite({center},{center_opp}).\n")
    file.write("\n")

    for center in clockwise.keys():
        centers_cl: List[str] = clockwise[center]
        for idx, center_cl in enumerate(centers_cl):
            idx_next = (idx + 1) % len(centers_cl)
            file.write(f"face_adjacent_dir({center}, {center_cl}, {centers_cl[idx_next]}, cl).\n")

    for center_opp in clockwise.keys():
        center: str = opposite[center_opp]
        centers_cl: List[str] = clockwise[center_opp]
        for idx, center_cl in enumerate(centers_cl):
            idx_next = (idx - 1) % len(centers_cl)
            file.write(f"face_adjacent_dir({center}, {center_cl}, {centers_cl[idx_next]}, cl).\n")
    file.write("\n")

    file.write("face_adjacent_dir(A,B,C,cc) :- face_adjacent_dir(A,C,B,cl).\n")
    file.write("\n")

    file.write("onface(state(A, _), w, A).\n")
    file.write("onface(state(_, A), g, A).\n")

    file.close()


if __name__ == '__main__':
    main()
