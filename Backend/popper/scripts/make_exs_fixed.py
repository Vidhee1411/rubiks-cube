from typing import List, Tuple


# pos(move(w_c, cl, w_c, g_c)).
# pos(move(w_c, cc, w_c, b_c)).

# pos(move(o_c, cl, b_c, o_c)).
# pos(move(o_c, cc, g_c, o_c)).


def main():
    file = open("examples/cube/exs.pl", "w")
    faces: List[str] = ['w_c', 'y_c', 'o_c', 'b_c', 'g_c', 'r_c']
    direcs: List[str] = ['cl', 'cc']
    pos_to_move: List[Tuple[str, str, str, str]] = [
        ("w_c", "cl", "w_c", "g_c"),
        ("w_c", "cc", "w_c", "b_c"),
        # ("w_c", "cc", "w_c", "r_c"),

        ("o_c", "cl", "b_c", "o_c"),
        ("o_c", "cc", "g_c", "o_c"),
        # ("o_c", "cc", "y_c", "o_c"),
    ]

    for face, direc, w_pos, o_pos in pos_to_move:
        file.write(f"pos(move({face}, {direc}, {w_pos}, {o_pos})).\n")
    file.write("\n")

    for direc_neg in direcs:
        for face_neg in faces:
            file.write(f"neg(move({face_neg}, {direc_neg}, w_c, o_c)).\n")
    file.write("\n")

    file.write("\n")
    for face, direc, w_pos, o_pos in pos_to_move:
        for direc_neg in direcs:
            for face_neg in faces:
                if (direc_neg == direc) and (face_neg == face):
                    continue

                file.write(f"neg(move({face_neg}, {direc_neg}, {w_pos}, {o_pos})).\n")

    file.close()


if __name__ == '__main__':
    main()
