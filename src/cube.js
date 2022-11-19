import * as THREE from "three";
import { Reflector } from "three/examples/jsm/objects/Reflector";

import { scene, recreate, topFace, bottomFace, rightFace, leftFace, frontFace, backFace, resetCameraPosition } from "./main";
import { applyState } from "./state";
import { matrixTransformationCW, matrixTransformationCCW, updateXWithAnimation, updateYWithAnimation, updateZWithAnimation } from "./utils";

export var outlineMesh;

var outlineMeshColor = 0x4deeea;
var red = 0xff0000;
var orange = 0xffa500;
var yellow = 0xffff00;
var white = 0xffffff;
var blue = 0x0000ff;
var green = 0x008000;
var black = 0x000000;
var gray = 0x808080;

//This will be used to store the entire cube wireframe in runtime
var line = null;

//Cube is stored in this
export var cubeArray = [];

var animationTime = 1000;

var sliderValue = document.getElementById("speedSlider");
sliderValue.addEventListener("change", function () {
  animationTime = 1000 / sliderValue.value;
});

export function disableSpeedSlider() {
  sliderValue.disabled = true;
}

export function enableSpeedSlider() {
  sliderValue.disabled = false;
}

export function getAnimationTime() {
  return animationTime;
}

var moving = false;
//Mirror1 - Right Mirror
//Mirror2 - Bottom Mirror
//Mirror3 = Left Mirror
var mirror1, mirror2, mirror3;

function updateStickerStateX(i, j, k, ccw = False) {
  var obj = { ...cubeArray[i][j][k][3] };

  if (ccw) {
    //Clockwise
    //Front -> Up -> Back -> Bottom -> Front
    cubeArray[i][j][k][3].frontFace = obj.topFace;
    cubeArray[i][j][k][3].topFace = obj.backFace;
    cubeArray[i][j][k][3].backFace = obj.bottomFace;
    cubeArray[i][j][k][3].bottomFace = obj.frontFace;
    return;
  }

  //C-Clockwise
  //Front -> Bottom -> Back -> Top -> Front
  cubeArray[i][j][k][3].frontFace = obj.bottomFace;
  cubeArray[i][j][k][3].bottomFace = obj.backFace;
  cubeArray[i][j][k][3].backFace = obj.topFace;
  cubeArray[i][j][k][3].topFace = obj.frontFace;
}

export function turnX(index, ccw = false) {
  if (moving) return;
  else {
    moving = true;
    setTimeout(() => {
      moving = false;
    }, getAnimationTime() * 1.3);
  }
  resetCameraPosition();
  var pivot = generateCubie();
  pivot.position.set(0, 0, 0);
  scene.add(pivot);
  for (var i = -1; i < 2; i++) {
    for (var j = -1; j < 2; j++) {
      for (var k = -1; k < 2; k++) {
        if (cubeArray[i][j][k][1].curX == index) {
          pivot.attach(cubeArray[i][j][k][0]);
          var matrixY = cubeArray[i][j][k][1].curY;
          var matrixZ = cubeArray[i][j][k][1].curZ;
          var transform;
          if (!ccw) {
            transform = matrixTransformationCW(matrixY, matrixZ);
          } else transform = matrixTransformationCCW(matrixY, matrixZ);
          //console.log(transform);
          cubeArray[i][j][k][1].curY = transform[0];
          cubeArray[i][j][k][1].curZ = transform[1];
          updateStickerStateX(i, j, k, ccw);
        }
      }
    }
  }
  makeOutline(1, 3, 3, index, 0, 0);
  if (!ccw) updateXWithAnimation(pivot, -1, getAnimationTime());
  else updateXWithAnimation(pivot, 1, getAnimationTime());
}

function updateStickerStateY(i, j, k, ccw = False) {
  var obj = { ...cubeArray[i][j][k][3] };

  if (!ccw) {
    //Clockwise
    //Front -> Left -> Back -> Right -> Front
    cubeArray[i][j][k][3].frontFace = obj.leftFace;
    cubeArray[i][j][k][3].leftFace = obj.backFace;
    cubeArray[i][j][k][3].backFace = obj.rightFace;
    cubeArray[i][j][k][3].rightFace = obj.frontFace;
    return;
  }

  //C-Clockwise
  //Front -> Right -> Back -> Left -> Front
  cubeArray[i][j][k][3].frontFace = obj.rightFace;
  cubeArray[i][j][k][3].rightFace = obj.backFace;
  cubeArray[i][j][k][3].backFace = obj.leftFace;
  cubeArray[i][j][k][3].leftFace = obj.frontFace;
}

export function turnY(index, ccw = false) {
  if (moving) return;
  else {
    moving = true;
    setTimeout(() => {
      moving = false;
    }, getAnimationTime() * 1.3);
  }
  resetCameraPosition();
  var pivot = generateCubie();
  pivot.position.set(0, 0, 0);
  pivot.rotation.x = -Math.PI;
  pivot.rotation.z = -Math.PI;
  scene.add(pivot);
  for (var i = -1; i < 2; i++) {
    for (var j = -1; j < 2; j++) {
      for (var k = -1; k < 2; k++) {
        if (cubeArray[i][j][k][1].curY == index) {
          pivot.attach(cubeArray[i][j][k][0]);
          var matrixX = cubeArray[i][j][k][1].curX;
          var matrixZ = cubeArray[i][j][k][1].curZ;
          var transform;
          if (!ccw) transform = matrixTransformationCW(matrixX, matrixZ);
          else transform = matrixTransformationCCW(matrixX, matrixZ);
          //console.log(transform);
          cubeArray[i][j][k][1].curX = transform[0];
          cubeArray[i][j][k][1].curZ = transform[1];
          updateStickerStateY(i, j, k, ccw);
        }
      }
    }
  }
  makeOutline(3, 1, 3, 0, index, 0);
  if (!ccw) updateYWithAnimation(pivot, -1, getAnimationTime());
  else updateYWithAnimation(pivot, 1, getAnimationTime());
}

function updateStickerStateZ(i, j, k, ccw = False) {
  var obj = { ...cubeArray[i][j][k][3] };

  if (ccw) {
    //Clockwise
    //Up -> Right -> Bottom -> Left -> Up
    cubeArray[i][j][k][3].topFace = obj.rightFace;
    cubeArray[i][j][k][3].rightFace = obj.bottomFace;
    cubeArray[i][j][k][3].bottomFace = obj.leftFace;
    cubeArray[i][j][k][3].leftFace = obj.topFace;
    return;
  }

  //C-Clockwise
  //Up -> Left -> Bottom -> Right -> Up
  cubeArray[i][j][k][3].topFace = obj.leftFace;
  cubeArray[i][j][k][3].leftFace = obj.bottomFace;
  cubeArray[i][j][k][3].bottomFace = obj.rightFace;
  cubeArray[i][j][k][3].rightFace = obj.topFace;
}

export function turnZ(index, ccw = false) {
  if (moving) return;
  else {
    moving = true;
    setTimeout(() => {
      moving = false;
    }, getAnimationTime() * 1.3);
  }
  resetCameraPosition();
  var pivot = generateCubie();
  pivot.position.set(0, 0, 0);
  scene.add(pivot);
  for (var i = -1; i < 2; i++) {
    for (var j = -1; j < 2; j++) {
      for (var k = -1; k < 2; k++) {
        if (cubeArray[i][j][k][1].curZ == index) {
          pivot.attach(cubeArray[i][j][k][0]);
          var matrixX = cubeArray[i][j][k][1].curX;
          var matrixY = cubeArray[i][j][k][1].curY;
          var transform;
          if (!ccw) transform = matrixTransformationCW(matrixX, matrixY);
          else transform = matrixTransformationCCW(matrixX, matrixY);
          //console.log(transform);
          cubeArray[i][j][k][1].curX = transform[0];
          cubeArray[i][j][k][1].curY = transform[1];
          updateStickerStateZ(i, j, k, ccw);
        }
      }
    }
  }
  makeOutline(3, 3, 1, 0, 0, index);
  if (!ccw) updateZWithAnimation(pivot, -1, getAnimationTime());
  else updateZWithAnimation(pivot, 1, getAnimationTime());
}

function removeOutline() {
  scene.remove(outlineMesh);
  line.visible = true;
}

async function makeOutline(sizeX, sizeY, sizeZ, x, y, z) {
  line.visible = false;
  var outlineMaterial = new THREE.MeshBasicMaterial({
    color: outlineMeshColor,
    side: THREE.BackSide,
  });
  outlineMesh = new THREE.Mesh(new THREE.BoxGeometry(sizeX, sizeY, sizeZ), outlineMaterial);
  outlineMesh.position.set(x, y, z);
  outlineMesh.scale.multiplyScalar(1.1);
  scene.add(outlineMesh);
  setTimeout(removeOutline, getAnimationTime());
}

var edgeCubeHighlight;

export function highlightEdgePiece() {
  edgeCubeHighlight = getOutline(1, 1, 1, 0, 1, 1);
  scene.add(edgeCubeHighlight);
}

export function removeEdgePieceHighlight() {
  scene.remove(edgeCubeHighlight);
}

var cornerPieceHighlight;

export function highlightCornerPiece() {
  cornerPieceHighlight = getOutline(1, 1, 1, 1, 1, 1);
  scene.add(cornerPieceHighlight);
}

export function removeCornerPieceHighlight() {
  scene.remove(cornerPieceHighlight);
}

var stickerHighlight;

export function highlightSticker() {
  stickerHighlight = getOutline(1, 1, 0.1, 0, 1, 1.5);
  scene.add(stickerHighlight);
}

export function removeStickerHighlight() {
  scene.remove(stickerHighlight);
}

var fullCubeHighlightMesh;

export async function makeHighlight(sizeX, sizeY, sizeZ, x, y, z, multiply) {
  //line.visible = false;
  var outlineMaterial = new THREE.MeshBasicMaterial({
    color: outlineMeshColor,
    side: THREE.BackSide,
  });
  fullCubeHighlightMesh = new THREE.Mesh(new THREE.BoxGeometry(sizeX, sizeY, sizeZ), outlineMaterial);
  fullCubeHighlightMesh.position.set(x, y, z);
  fullCubeHighlightMesh.scale.multiplyScalar(multiply);
  scene.add(fullCubeHighlightMesh);
}

export function removeFullCubeHighlight() {
  scene.remove(fullCubeHighlightMesh);
}

export function generateCube() {
  var wireframeSize = 2.97;
  const geometry = new THREE.BoxGeometry(wireframeSize, wireframeSize, wireframeSize);
  const edges = new THREE.EdgesGeometry(geometry);
  line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x000000 }));
  line.position.set(0, 0, 0);
  scene.add(line);

  for (var i = -1; i < 2; i++) {
    cubeArray[i] = [];
    for (var j = -1; j < 2; j++) {
      cubeArray[i][j] = [];
      for (var k = -1; k < 2; k++) {
        cubeArray[i][j][k] = [];
        var cubie = generateCubie();
        cubie.position.set(i, j, k);
        cubeArray[i][j][k][0] = cubie;
        cubeArray[i][j][k][1] = { curX: i, curY: j, curZ: k };
        cubeArray[i][j][k][2] = {
          topFace: black,
          bottomFace: black,
          rightFace: black,
          leftFace: black,
          frontFace: black,
          backFace: black,
        };
        cubeArray[i][j][k][3] = {
          topFace: topFace,
          bottomFace: bottomFace,
          rightFace: rightFace,
          leftFace: leftFace,
          frontFace: frontFace,
          backFace: backFace,
        };
        for (var l = 0; l < cubeArray[i][j][k][0].geometry.faces.length; l++) {
          cubeArray[i][j][k][0].geometry.faces[l].color.setHex(black);
        }
        scene.add(cubeArray[i][j][k][0]);
      }
    }
  }
}

/*
Cube face indexes
Right index 0, 1
Left index 2, 3
Top index 4, 5
Bottom index 6, 7
Front index 8, 9
Back index 10, 11
*/
export function generateCubie() {
  var geometry = new THREE.BoxGeometry(0.95, 0.95, 0.95);
  geometry.colorsNeedUpdate = true;
  var material = new THREE.MeshBasicMaterial({
    color: 0xffffff,
    vertexColors: true,
  });
  var cubie = new THREE.Mesh(geometry, material);
  return cubie;
}

var frontFaceHighlight;
var rightFaceHighlight;
var topFaceHighlight;
export function highlightFrontFace() {
  frontFaceHighlight = getOutline(3, 3, 1, 0, 0, 1);
  scene.add(frontFaceHighlight);
}
export function removeFrontFaceHighlight() {
  scene.remove(frontFaceHighlight);
}
export function highlightRightFace() {
  rightFaceHighlight = getOutline(1, 3, 3, 1, 0, 0);
  scene.add(rightFaceHighlight);
}
export function removeRightFaceHighlight() {
  scene.remove(rightFaceHighlight);
}
export function highlightTopFace() {
  topFaceHighlight = getOutline(3, 1, 3, 0, 1, 0);
  scene.add(topFaceHighlight);
}
export function removeTopFaceHighlight() {
  scene.remove(topFaceHighlight);
}

var mirror1Highlight;
var mirror2Highlight;
var mirror3Highlight;
export function getOutline(sizeX, sizeY, sizeZ, x, y, z, flip = 0) {
  var outlineMaterial = new THREE.MeshBasicMaterial({
    color: outlineMeshColor,
    side: THREE.BackSide,
  });
  var mesh = new THREE.Mesh(new THREE.BoxGeometry(sizeX, sizeY, sizeZ), outlineMaterial);
  mesh.position.set(x, y, z);
  mesh.scale.multiplyScalar(1.1);
  if (flip == 1) {
    mesh.rotation.y = -Math.PI / 2;
    return mesh;
  }
  if (flip == 2) {
    mesh.rotation.y = Math.PI / 2;
    return mesh;
  }
  return mesh;
}
export function highlightMirror1() {
  mirror1Highlight = getOutline(0.1, 2.3, 2.3, -6.2, 1.7, 1.7);
  scene.add(mirror1Highlight);
}
export function removeMirror1Highlight() {
  scene.remove(mirror1Highlight);
}
export function highlightMirror2() {
  mirror2Highlight = getOutline(2.3, 2.3, 0.1, 1.7, 1.7, -6.2, 0);
  scene.add(mirror2Highlight);
}
export function removeMirror2Highlight() {
  scene.remove(mirror2Highlight);
}
export function highlightMirror3() {
  mirror3Highlight = getOutline(2.3, 0.1, 2.3, 1.7, -6.2, 1.7, 0);
  scene.add(mirror3Highlight);
}
export function removeMirror3Highlight() {
  scene.remove(mirror3Highlight);
}

export function generateAllMirrors() {
  mirror1 = generateMirror(6, 1.725, 1.725, 0);
  mirror2 = generateMirror(-1.725, -6, 1.725, 1);
  mirror3 = generateMirror(-1.725, 1.725, -6, 2);
  scene.add(mirror1);
  scene.add(mirror2);
  scene.add(mirror3);
}

export function deactivateMirrors() {
  mirror1.visible = false;
  mirror2.visible = false;
  mirror3.visible = false;
}
export function activateMirrors() {
  mirror1.visible = true;
  mirror2.visible = true;
  mirror3.visible = true;
}

/*
  Position of mirror (x, y, z)
  And provide an extra parameter (0, 1, 2) for flipping the mirrors
  0 for back side mirror
  1 for bottom mirror
  2 for left mirror
*/
function generateMirror(x, y, z, flip) {
  var mirror = new Reflector(new THREE.PlaneBufferGeometry(2, 2), {
    textureWidth: window.innerWidth * window.devicePixelRatio,
    textureHeight: window.innerHeight * window.devicePixelRatio,
  });
  mirror.position.z -= x;
  mirror.position.y += y;
  mirror.position.x += z;
  if (flip == 1) {
    mirror.rotation.x = -Math.PI / 2;
    return mirror;
  }
  if (flip == 2) {
    mirror.rotation.y = Math.PI / 2;
    return mirror;
  }
  return mirror;
}

export function grayCubieAtXYZOnFaces(x, y, z, faces) {
  for (var i = -1; i < 2; i++) {
    for (var j = -1; j < 2; j++) {
      for (var k = -1; k < 2; k++) {
        if (cubeArray[i][j][k][1].curX == x && cubeArray[i][j][k][1].curY == y && cubeArray[i][j][k][1].curZ == z) {
          cubeArray[i][j][k][0].geometry.colorsNeedUpdate = true;
          faces.forEach((n) => {
            cubeArray[i][j][k][0].geometry.faces[n].color.setHex(gray);
          });
        }
      }
    }
  }
}

// export function changeColorOfFace(i, j, k, color, face) {
//   cubeArray[i][j][k][0].geometry.colorsNeedUpdate = true;
//   cubeArray[i][j][k][0].geometry.faces[face[0]].color.setHex(color);
//   cubeArray[i][j][k][0].geometry.faces[face[1]].color.setHex(color);
// }

export function changeColorCurrentState(x, y, z, color, face) {
  for (var i = -1; i < 2; i++) {
    for (var j = -1; j < 2; j++) {
      for (var k = -1; k < 2; k++) {
        if (cubeArray[i][j][k][1].curX == x && cubeArray[i][j][k][1].curY == y && cubeArray[i][j][k][1].curZ == z) {
          cubeArray[i][j][k][0].geometry.colorsNeedUpdate = true;
          cubeArray[i][j][k][3][face].forEach((n) => {
            cubeArray[i][j][k][0].geometry.faces[n].color.setHex(color);
          });
        }
      }
    }
  }
}

export function assignColors() {
  assignRed();
  assignOrange();
  assignWhite();
  assignYellow();
  assignGreen();
  assignBlue();
}

function assignRed() {
  for (var i = -1; i < 2; i++) {
    for (var j = -1; j < 2; j++) {
      for (var k = -1; k < 2; k++) {
        if (i == 1) {
          cubeArray[i][j][k][0].geometry.faces[0].color.setHex(red);
          cubeArray[i][j][k][0].geometry.faces[1].color.setHex(red);
          cubeArray[i][j][k][2].rightFace = red;
        }
      }
    }
  }
}

function assignOrange() {
  for (var i = -1; i < 2; i++) {
    for (var j = -1; j < 2; j++) {
      for (var k = -1; k < 2; k++) {
        if (i == -1) {
          cubeArray[i][j][k][0].geometry.faces[2].color.setHex(orange);
          cubeArray[i][j][k][0].geometry.faces[3].color.setHex(orange);
          cubeArray[i][j][k][2].leftFace = orange;
        }
      }
    }
  }
}

function assignWhite() {
  for (var i = -1; i < 2; i++) {
    for (var j = -1; j < 2; j++) {
      for (var k = -1; k < 2; k++) {
        if (j == 1) {
          cubeArray[i][j][k][0].geometry.faces[4].color.setHex(white);
          cubeArray[i][j][k][0].geometry.faces[5].color.setHex(white);
          cubeArray[i][j][k][2].topFace = white;
        }
      }
    }
  }
}

function assignYellow() {
  for (var i = -1; i < 2; i++) {
    for (var j = -1; j < 2; j++) {
      for (var k = -1; k < 2; k++) {
        if (j == -1) {
          cubeArray[i][j][k][0].geometry.faces[6].color.setHex(yellow);
          cubeArray[i][j][k][0].geometry.faces[7].color.setHex(yellow);
          cubeArray[i][j][k][2].bottomFace = yellow;
        }
      }
    }
  }
}

function assignGreen() {
  for (var i = -1; i < 2; i++) {
    for (var j = -1; j < 2; j++) {
      for (var k = -1; k < 2; k++) {
        if (k == 1) {
          cubeArray[i][j][k][0].geometry.faces[8].color.setHex(green);
          cubeArray[i][j][k][0].geometry.faces[9].color.setHex(green);
          cubeArray[i][j][k][2].frontFace = green;
        }
      }
    }
  }
}

function assignBlue() {
  for (var i = -1; i < 2; i++) {
    for (var j = -1; j < 2; j++) {
      for (var k = -1; k < 2; k++) {
        if (k == -1) {
          cubeArray[i][j][k][0].geometry.faces[10].color.setHex(blue);
          cubeArray[i][j][k][0].geometry.faces[11].color.setHex(blue);
          cubeArray[i][j][k][2].backFace = blue;
        }
      }
    }
  }
}

export function resetToOriginalColors() {
  for (var i = -1; i < 2; i++) {
    for (var j = -1; j < 2; j++) {
      for (var k = -1; k < 2; k++) {
        cubeArray[i][j][k][0].geometry.colorsNeedUpdate = true;
        cubeArray[i][j][k][0].geometry.faces[0].color.setHex(cubeArray[i][j][k][2].rightFace);
        cubeArray[i][j][k][0].geometry.faces[1].color.setHex(cubeArray[i][j][k][2].rightFace);
        cubeArray[i][j][k][0].geometry.faces[2].color.setHex(cubeArray[i][j][k][2].leftFace);
        cubeArray[i][j][k][0].geometry.faces[3].color.setHex(cubeArray[i][j][k][2].leftFace);
        cubeArray[i][j][k][0].geometry.faces[4].color.setHex(cubeArray[i][j][k][2].topFace);
        cubeArray[i][j][k][0].geometry.faces[5].color.setHex(cubeArray[i][j][k][2].topFace);
        cubeArray[i][j][k][0].geometry.faces[6].color.setHex(cubeArray[i][j][k][2].bottomFace);
        cubeArray[i][j][k][0].geometry.faces[7].color.setHex(cubeArray[i][j][k][2].bottomFace);
        cubeArray[i][j][k][0].geometry.faces[8].color.setHex(cubeArray[i][j][k][2].frontFace);
        cubeArray[i][j][k][0].geometry.faces[9].color.setHex(cubeArray[i][j][k][2].frontFace);
        cubeArray[i][j][k][0].geometry.faces[10].color.setHex(cubeArray[i][j][k][2].backFace);
        cubeArray[i][j][k][0].geometry.faces[11].color.setHex(cubeArray[i][j][k][2].backFace);
      }
    }
  }
}

export function resetScene() {
  scene.remove.apply(scene, scene.children);
  recreate();
}

export function resetCubeConfiguration() {
  applyState([
      1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 18, 18, 18, 18, 18, 18, 18, 18, 18, 27, 27, 27, 27, 27, 27, 27, 27, 27, 36, 36, 36, 36, 36, 36, 36,
      36, 36, 45, 45, 45, 45, 45, 45, 45, 45, 45,
  ]);
}
