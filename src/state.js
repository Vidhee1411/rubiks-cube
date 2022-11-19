import { changeColorCurrentState, changeColorOfFace, cubeArray } from "./cube";
import { frontFace, backFace, rightFace, leftFace, topFace, bottomFace, getCurrentState, updateSolveState } from "./main";

/*
  Getting an array object of 54 length
  which would contain sticker colors
  Considering the format of state parameter would be
  [                                                                                 - directions from the initial point of view
    9 top stickers(with white center piece) 0, 1, 2, 3, 4, 5, 6, 7, 8               - starting from topLeft towards right
    9 bottom stickers(with yellow center piece) 9, 10, 11, 12, 13, 14, 15, 16, 17   - starting from topRight towards left
    9 left stickers(with orange center piece) 18, 19, 20, 21, 22, 23, 24, 25, 26    - starting from bottomRight going up
    9 right stickers(with red center piece) 27, 28, 29, 30, 31, 32, 33, 34, 35      - starting from bottomLeft going up
    9 back stickers(with blue center piece) 36, 37, 38, 39, 40, 41, 42, 43, 44      - starting from bottomRight going up
    9 front stickers(with green center piece) 45, 46, 47, 48, 49, 50, 51, 52, 53    - starting from frontRight going up
  ]
*/

var red = 0xff0000;
var orange = 0xffa500;
var yellow = 0xffff00;
var white = 0xffffff;
var blue = 0x0000ff;
var green = 0x008000;
var black = 0x000000;
var gray = 0x808080;

var current_state;

export function applyState(state_colors) {
  if (state_colors == null || state_colors.length != 54) return;
  changeColorCurrentState(-1, 1, 1, findColor(state_colors[0]), "topFace");
  //White center piece
  changeColorCurrentState(-1, 1, 1, findColor(state_colors[0]), "topFace");
  changeColorCurrentState(-1, 1, 0, findColor(state_colors[1]), "topFace");
  changeColorCurrentState(-1, 1, -1, findColor(state_colors[2]), "topFace");
  changeColorCurrentState(0, 1, 1, findColor(state_colors[3]), "topFace");
  changeColorCurrentState(0, 1, 0, findColor(state_colors[4]), "topFace");
  changeColorCurrentState(0, 1, -1, findColor(state_colors[5]), "topFace");
  changeColorCurrentState(1, 1, 1, findColor(state_colors[6]), "topFace");
  changeColorCurrentState(1, 1, 0, findColor(state_colors[7]), "topFace");
  changeColorCurrentState(1, 1, -1, findColor(state_colors[8]), "topFace");

  //Yellow center piece
  changeColorCurrentState(-1, -1, -1, findColor(state_colors[9]), "bottomFace");
  changeColorCurrentState(-1, -1, 0, findColor(state_colors[10]), "bottomFace");
  changeColorCurrentState(-1, -1, 1, findColor(state_colors[11]), "bottomFace");
  changeColorCurrentState(0, -1, -1, findColor(state_colors[12]), "bottomFace");
  changeColorCurrentState(0, -1, 0, findColor(state_colors[13]), "bottomFace");
  changeColorCurrentState(0, -1, 1, findColor(state_colors[14]), "bottomFace");
  changeColorCurrentState(1, -1, -1, findColor(state_colors[15]), "bottomFace");
  changeColorCurrentState(1, -1, 0, findColor(state_colors[16]), "bottomFace");
  changeColorCurrentState(1, -1, 1, findColor(state_colors[17]), "bottomFace");

  //Orange center piece
  changeColorCurrentState(-1, -1, -1, findColor(state_colors[18]), "leftFace");
  changeColorCurrentState(-1, 0, -1, findColor(state_colors[19]), "leftFace");
  changeColorCurrentState(-1, 1, -1, findColor(state_colors[20]), "leftFace");
  changeColorCurrentState(-1, -1, 0, findColor(state_colors[21]), "leftFace");
  changeColorCurrentState(-1, 0, 0, findColor(state_colors[22]), "leftFace");
  changeColorCurrentState(-1, 1, 0, findColor(state_colors[23]), "leftFace");
  changeColorCurrentState(-1, -1, 1, findColor(state_colors[24]), "leftFace");
  changeColorCurrentState(-1, 0, 1, findColor(state_colors[25]), "leftFace");
  changeColorCurrentState(-1, 1, 1, findColor(state_colors[26]), "leftFace");

  //Red center piece
  changeColorCurrentState(1, -1, 1, findColor(state_colors[27]), "rightFace");
  changeColorCurrentState(1, 0, 1, findColor(state_colors[28]), "rightFace");
  changeColorCurrentState(1, 1, 1, findColor(state_colors[29]), "rightFace");
  changeColorCurrentState(1, -1, 0, findColor(state_colors[30]), "rightFace");
  changeColorCurrentState(1, 0, 0, findColor(state_colors[31]), "rightFace");
  changeColorCurrentState(1, 1, 0, findColor(state_colors[32]), "rightFace");
  changeColorCurrentState(1, -1, -1, findColor(state_colors[33]), "rightFace");
  changeColorCurrentState(1, 0, -1, findColor(state_colors[34]), "rightFace");
  changeColorCurrentState(1, 1, -1, findColor(state_colors[35]), "rightFace");

  //Blue center piece
  changeColorCurrentState(1, -1, -1, findColor(state_colors[36]), "backFace");
  changeColorCurrentState(1, 0, -1, findColor(state_colors[37]), "backFace");
  changeColorCurrentState(1, 1, -1, findColor(state_colors[38]), "backFace");
  changeColorCurrentState(0, -1, -1, findColor(state_colors[39]), "backFace");
  changeColorCurrentState(0, 0, -1, findColor(state_colors[40]), "backFace");
  changeColorCurrentState(0, 1, -1, findColor(state_colors[41]), "backFace");
  changeColorCurrentState(-1, -1, -1, findColor(state_colors[42]), "backFace");
  changeColorCurrentState(-1, 0, -1, findColor(state_colors[43]), "backFace");
  changeColorCurrentState(-1, 1, -1, findColor(state_colors[44]), "backFace");

  //Green center piece
  changeColorCurrentState(-1, -1, 1, findColor(state_colors[45]), "frontFace");
  changeColorCurrentState(-1, 0, 1, findColor(state_colors[46]), "frontFace");
  changeColorCurrentState(-1, 1, 1, findColor(state_colors[47]), "frontFace");
  changeColorCurrentState(0, -1, 1, findColor(state_colors[48]), "frontFace");
  changeColorCurrentState(0, 0, 1, findColor(state_colors[49]), "frontFace");
  changeColorCurrentState(0, 1, 1, findColor(state_colors[50]), "frontFace");
  changeColorCurrentState(1, -1, 1, findColor(state_colors[51]), "frontFace");
  changeColorCurrentState(1, 0, 1, findColor(state_colors[52]), "frontFace");
  changeColorCurrentState(1, 1, 1, findColor(state_colors[53]), "frontFace");

  //updateState('', true);
}

// export function applyStateIfInitial(state_colors) {
//   //White center piece
//   changeColorOfFace(-1, 1, 1, findColor(state_colors[0]), topFace);
//   changeColorOfFace(-1, 1, 0, findColor(state_colors[1]), topFace);
//   changeColorOfFace(-1, 1, -1, findColor(state_colors[2]), topFace);
//   changeColorOfFace(0, 1, 1, findColor(state_colors[3]), topFace);
//   changeColorOfFace(0, 1, 0, findColor(state_colors[4]), topFace);
//   changeColorOfFace(0, 1, -1, findColor(state_colors[5]), topFace);
//   changeColorOfFace(1, 1, 1, findColor(state_colors[6]), topFace);
//   changeColorOfFace(1, 1, 0, findColor(state_colors[7]), topFace);
//   changeColorOfFace(1, 1, -1, findColor(state_colors[8]), topFace);
//   //Yellow center piece
//   changeColorOfFace(-1, -1, -1, findColor(state_colors[9]), bottomFace);
//   changeColorOfFace(-1, -1, 0, findColor(state_colors[10]), bottomFace);
//   changeColorOfFace(-1, -1, 1, findColor(state_colors[11]), bottomFace);
//   changeColorOfFace(0, -1, -1, findColor(state_colors[12]), bottomFace);
//   changeColorOfFace(0, -1, 0, findColor(state_colors[13]), bottomFace);
//   changeColorOfFace(0, -1, 1, findColor(state_colors[14]), bottomFace);
//   changeColorOfFace(1, -1, -1, findColor(state_colors[15]), bottomFace);
//   changeColorOfFace(1, -1, 0, findColor(state_colors[16]), bottomFace);
//   changeColorOfFace(1, -1, 1, findColor(state_colors[17]), bottomFace);
//   //Orange center piece
//   changeColorOfFace(-1, -1, -1, findColor(state_colors[18]), leftFace);
//   changeColorOfFace(-1, 0, -1, findColor(state_colors[19]), leftFace);
//   changeColorOfFace(-1, 1, -1, findColor(state_colors[20]), leftFace);
//   changeColorOfFace(-1, -1, 0, findColor(state_colors[21]), leftFace);
//   changeColorOfFace(-1, 0, 0, findColor(state_colors[22]), leftFace);
//   changeColorOfFace(-1, 1, 0, findColor(state_colors[23]), leftFace);
//   changeColorOfFace(-1, -1, 1, findColor(state_colors[24]), leftFace);
//   changeColorOfFace(-1, 0, 1, findColor(state_colors[25]), leftFace);
//   changeColorOfFace(-1, 1, 1, findColor(state_colors[26]), leftFace);
//   //Red center piece
//   changeColorOfFace(1, -1, 1, findColor(state_colors[27]), rightFace);
//   changeColorOfFace(1, 0, 1, findColor(state_colors[28]), rightFace);
//   changeColorOfFace(1, 1, 1, findColor(state_colors[29]), rightFace);
//   changeColorOfFace(1, -1, 0, findColor(state_colors[30]), rightFace);
//   changeColorOfFace(1, 0, 0, findColor(state_colors[31]), rightFace);
//   changeColorOfFace(1, 1, 0, findColor(state_colors[32]), rightFace);
//   changeColorOfFace(1, -1, -1, findColor(state_colors[33]), rightFace);
//   changeColorOfFace(1, 0, -1, findColor(state_colors[34]), rightFace);
//   changeColorOfFace(1, 1, -1, findColor(state_colors[35]), rightFace);
//   //Blue center piece
//   changeColorOfFace(1, -1, -1, findColor(state_colors[36]), backFace);
//   changeColorOfFace(1, 0, -1, findColor(state_colors[37]), backFace);
//   changeColorOfFace(1, 1, -1, findColor(state_colors[38]), backFace);
//   changeColorOfFace(0, -1, -1, findColor(state_colors[39]), backFace);
//   changeColorOfFace(0, 0, -1, findColor(state_colors[40]), backFace);
//   changeColorOfFace(0, 1, -1, findColor(state_colors[41]), backFace);
//   changeColorOfFace(-1, -1, -1, findColor(state_colors[42]), backFace);
//   changeColorOfFace(-1, 0, -1, findColor(state_colors[43]), backFace);
//   changeColorOfFace(-1, 1, -1, findColor(state_colors[44]), backFace);
//   //Green center piece
//   changeColorOfFace(-1, -1, 1, findColor(state_colors[45]), frontFace);
//   changeColorOfFace(-1, 0, 1, findColor(state_colors[46]), frontFace);
//   changeColorOfFace(-1, 1, 1, findColor(state_colors[47]), frontFace);
//   changeColorOfFace(0, -1, 1, findColor(state_colors[48]), frontFace);
//   changeColorOfFace(0, 0, 1, findColor(state_colors[49]), frontFace);
//   changeColorOfFace(0, 1, 1, findColor(state_colors[50]), frontFace);
//   changeColorOfFace(1, -1, 1, findColor(state_colors[51]), frontFace);
//   changeColorOfFace(1, 0, 1, findColor(state_colors[52]), frontFace);
//   changeColorOfFace(1, 1, 1, findColor(state_colors[53]), frontFace);
// }

function findColor(num) {
  if (num < 0) {
    return gray;
  } else if (num < 9) {
    return white;
  } else if (num < 18) {
    return yellow;
  } else if (num < 27) {
    return orange;
  } else if (num < 36) {
    return red;
  } else if (num < 45) {
    return blue;
  } else if (num < 54) {
    return green;
  } else {
    return gray;
  }
}

export function updateState(move, update_config) {
    if (update_config) {
        current_state = getCurrentState()
    }
    
    console.log("Current ", current_state);
    //console.log("getCurrent ", getCurrentState());
    
    switch (move) {
        //Front face clockwise
        case "f":
            //Adjusting cubelets of four active faces
            rotateCubelets(0, 24, 17, 29);  //First cubelet
            rotateCubelets(3, 25, 14, 28);  //Second cubelet
            rotateCubelets(6, 26, 11, 27);  //Third cubelet
            
            //Adjusting cubelets of moving face - Front face
            rotateMovingFace([[47,50,53],[46,49,52],[45,48,51]], false);

            break;

        //Front face counter-clockwise
        case "f'":
            //Adjusting cubelets of four active faces
            rotateCubelets(29, 17, 24, 0);  //First cubelet
            rotateCubelets(28, 14, 25, 3);  //Second cubelet
            rotateCubelets(27, 11, 26, 6);  //Third cubelet

            //Adjusting cubelets of moving face - Front face
            rotateMovingFace([[47, 50, 53], [46, 49, 52], [45, 48, 51]], true);

            break;

        //Right face clockwise
        case "r":
            //Adjusting cubelets of four active faces
            rotateCubelets(51, 15, 38, 6);  //First cubelet
            rotateCubelets(52, 16, 37, 7);  //Second cubelet
            rotateCubelets(53, 17, 36, 8);  //Third cubelet

            //Adjusting cubelets of moving face - Right face
            rotateMovingFace([[29, 32, 35], [28, 31, 34], [27, 30, 33]], false);
            
            break;

        //Right face counter-clockwise
        case "r'":
            //Adjusting cubelets of four active faces
            rotateCubelets(6, 38, 15, 51);  //First cubelet
            rotateCubelets(7, 37, 16, 52);  //Second cubelet
            rotateCubelets(8, 36, 17, 53);  //Third cubelet

            //Adjusting cubelets of moving face - Right face
            rotateMovingFace([[29, 32, 35], [28, 31, 34], [27, 30, 33]], true);

            break;

        //Upper face clockwise
        case "u":
            //Adjusting cubelets of four active faces
            rotateCubelets(29, 38, 20, 47);  //First cubelet
            rotateCubelets(32, 41, 23, 50);  //Second cubelet
            rotateCubelets(35, 44, 26, 53);  //Third cubelet

            //Adjusting cubelets of moving face - Upper face
            rotateMovingFace([[0, 1, 2], [3, 4, 5], [6, 7, 8]], false);

            break;

        //Upper face counter-clockwise
        case "u'":
            //Adjusting cubelets of four active faces
            rotateCubelets(47, 20, 38, 29);  //First cubelet
            rotateCubelets(50, 23, 41, 32);  //Second cubelet
            rotateCubelets(53, 26, 44, 35);  //Third cubelet

            //Adjusting cubelets of moving face - Upper face
            rotateMovingFace([[0, 1, 2], [3, 4, 5], [6, 7, 8]], true);

            break;

        //Back face clockwise
        case "b":
            //Adjusting cubelets of four active faces
            rotateCubelets(35, 15, 18, 2);  //First cubelet
            rotateCubelets(34, 12, 19, 5);  //Second cubelet
            rotateCubelets(33, 9, 20, 8);  //Third cubelet

            //Adjusting cubelets of moving face - Back face
            rotateMovingFace([[38, 41, 44], [37, 40, 43], [36, 39, 42]], false);

            break;

        //Back face counter-clockwise
        case "b'":
            //Adjusting cubelets of four active faces
            rotateCubelets(2, 18, 15, 35);  //First cubelet
            rotateCubelets(5, 19, 12, 34);  //Second cubelet
            rotateCubelets(8, 20, 9, 33);  //Third cubelet

            //Adjusting cubelets of moving face - Back face
            rotateMovingFace([[38, 41, 44], [37, 40, 43], [36, 39, 42]], true);
            
            break;

        //Left face clockwise
        case "l":
            //Adjusting cubelets of four active faces
            rotateCubelets(0, 44, 9, 45);  //First cubelet
            rotateCubelets(1, 43, 10, 46);  //Second cubelet
            rotateCubelets(2, 42, 11, 47);  //Third cubelet

            //Adjusting cubelets of moving face - Left face
            rotateMovingFace([[26, 23, 20], [25, 22, 19], [24, 21, 18]], true);     //Exception -> Doesn't work similar to the other faces

            break;

        //Left face counter-clockwise
        case "l'":
            //Adjusting cubelets of four active faces
            rotateCubelets(45, 9, 44, 0);  //First cubelet
            rotateCubelets(46, 10, 43, 1);  //Second cubelet
            rotateCubelets(47, 11, 42, 2);  //Third cubelet

            //Adjusting cubelets of moving face - Left face
            rotateMovingFace([[26, 23, 20], [25, 22, 19], [24, 21, 18]], false);    //Exception -> Doesn't work similar to the other faces

            break;

        //Bottom face clockwise
        case "d":
            //Adjusting cubelets of four active faces
            rotateCubelets(27, 45, 18, 36);  //First cubelet
            rotateCubelets(30, 48, 21, 39);  //Second cubelet
            rotateCubelets(33, 51, 24, 42);  //Third cubelet

            //Adjusting cubelets of moving face - Bottom face
            rotateMovingFace([[9, 12, 15], [10, 13, 16], [11, 14, 17]], true);    //Exception -> Doesn't work similar to the other faces

            break;

        //Bottom face counter-clockwise
        case "d'":
            //Adjusting cubelets of four active faces
            rotateCubelets(36, 18, 45, 27);  //First cubelet
            rotateCubelets(39, 21, 48, 30);  //Second cubelet
            rotateCubelets(42, 24, 51, 33);  //Third cubelet

            //Adjusting cubelets of moving face - Bottom face
            rotateMovingFace([[9, 12, 15], [10, 13, 16], [11, 14, 17]], false);    //Exception -> Doesn't work similar to the other faces

            break;

        default:
            break;
    }

    if (move != '') {
        checkGoalState();
    }
}

function rotateCubelets(a, b, c, d) {
    var temp_cubelet;

    temp_cubelet = current_state[a]
    current_state[a] = current_state[b]
    current_state[b] = current_state[c]
    current_state[c] = current_state[d]
    current_state[d] = temp_cubelet
}

function rotateMovingFace(matrix, ccw) {
    if (!ccw) {
        rotateCubelets(matrix[0][1], matrix[1][0], matrix[2][1], matrix[1][2]);
        rotateCubelets(matrix[0][0], matrix[2][0], matrix[2][2], matrix[0][2]);
    }
    else {
        rotateCubelets(matrix[1][2], matrix[2][1], matrix[1][0], matrix[0][1]);
        rotateCubelets(matrix[0][2], matrix[2][2], matrix[2][0], matrix[0][0]);
    }
}

function checkGoalState() {
    let flag = false;

    //Checking for white-cross only for now, we should use a dictionary of goal states going forward
    for (let idx = 0; idx < 9; idx++) {
        if (idx % 2 != 0 || idx == 4) {
            //If any of the 5 cubelets in the cross aren't white, break the loop
            if (current_state[idx] != 1) {
                flag = true;
                break;
            }
        }
    }

    //Update solve state in firebase
    if (!flag) {
        updateSolveState();
    }
}