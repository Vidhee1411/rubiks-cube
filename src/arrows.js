import { scene } from "./main";
import * as THREE from "three";
import frontImageCW from "./images/arrowFrontCW.png";
import frontImageCCW from "./images/arrowFrontCCW.png";

import upDown from "./images/updown.png";
var frontC, frontCC, backC, backCC, leftC, leftCC, rightC, rightCC, upC, upCC, downC, downCC;

export function addArrows() {
  //FRONT CLOCKWISE ARROW
  var img = new THREE.MeshBasicMaterial({
    //CHANGED to MeshBasicMaterial
    map: THREE.ImageUtils.loadTexture(frontImageCW),
    transparent: true,
  });
  frontC = new THREE.Mesh(new THREE.PlaneGeometry(4, 4), img);
  frontC.overdraw = true;
  frontC.position.set(-1, 1, 1);
  scene.add(frontC);

  //FRONT COUNTER CLOCKWISE ARROW
  var img = new THREE.MeshBasicMaterial({
    //CHANGED to MeshBasicMaterial
    map: THREE.ImageUtils.loadTexture(frontImageCCW),
    transparent: true,
  });
  frontCC = new THREE.Mesh(new THREE.PlaneGeometry(4, 4), img);
  frontCC.overdraw = true;
  frontCC.position.set(1, 1, 1);
  scene.add(frontCC);

  //BACK CLOCKWISE ARROW
  var img = new THREE.MeshBasicMaterial({
    //CHANGED to MeshBasicMaterial
    map: THREE.ImageUtils.loadTexture(frontImageCCW),
    transparent: true,
  });
  backC = new THREE.Mesh(new THREE.PlaneGeometry(4, 4), img);
  backC.overdraw = true;
  backC.position.set(1, 1, -1);
  scene.add(backC);

  //BACK COUNTER CLOCKWISE ARROW
  var img = new THREE.MeshBasicMaterial({
    //CHANGED to MeshBasicMaterial
    map: THREE.ImageUtils.loadTexture(frontImageCW),
    transparent: true,
  });
  backCC = new THREE.Mesh(new THREE.PlaneGeometry(4, 4), img);
  backCC.overdraw = true;
  backCC.position.set(-1, 1, -1);
  scene.add(backCC);

  //LEFT CLOCKWISE ARROW
  var img = new THREE.MeshBasicMaterial({
    //CHANGED to MeshBasicMaterial
    map: THREE.ImageUtils.loadTexture(frontImageCCW),
    transparent: true,
  });
  leftC = new THREE.Mesh(new THREE.PlaneGeometry(4, 4), img);
  leftC.overdraw = true;
  leftC.rotation.y = Math.PI / 2;
  leftC.position.set(-1, 1, -1);
  scene.add(leftC);

  //LEFT COUNTER CLOCKWISE ARROW
  var img = new THREE.MeshBasicMaterial({
    //CHANGED to MeshBasicMaterial
    map: THREE.ImageUtils.loadTexture(frontImageCW),
    transparent: true,
  });
  leftCC = new THREE.Mesh(new THREE.PlaneGeometry(4, 4), img);
  leftCC.overdraw = true;
  leftCC.rotation.y = Math.PI / 2;
  leftCC.position.set(-1, 1, 1);
  scene.add(leftCC);

  //RIGHT CLOCKWISE ARROW
  var img = new THREE.MeshBasicMaterial({
    //CHANGED to MeshBasicMaterial
    map: THREE.ImageUtils.loadTexture(frontImageCW),
    transparent: true,
  });
  rightC = new THREE.Mesh(new THREE.PlaneGeometry(4, 4), img);
  rightC.overdraw = true;
  rightC.rotation.y = Math.PI / 2;
  rightC.position.set(1, 1, 1);
  scene.add(rightC);

  //RIGHT COUNTER CLOCKWISE ARROW
  var img = new THREE.MeshBasicMaterial({
    //CHANGED to MeshBasicMaterial
    map: THREE.ImageUtils.loadTexture(frontImageCCW),
    transparent: true,
  });
  rightCC = new THREE.Mesh(new THREE.PlaneGeometry(4, 4), img);
  rightCC.overdraw = true;
  rightCC.rotation.y = Math.PI / 2;
  rightCC.position.set(1, 1, -1);
  scene.add(rightCC);

  //UP CLOCKWISE ARROW
  var img = new THREE.MeshBasicMaterial({
    //CHANGED to MeshBasicMaterial
    map: THREE.ImageUtils.loadTexture(frontImageCW),
    transparent: true,
  });
  upC = new THREE.Mesh(new THREE.PlaneGeometry(4, 4), img);
  upC.overdraw = true;
  upC.rotation.x = -Math.PI / 2;
  upC.rotation.z = -Math.PI / 2;
  upC.position.set(0.8, 1, 0);
  scene.add(upC);

  //UP COUNTER CLOCKWISE ARROW
  var img = new THREE.MeshBasicMaterial({
    //CHANGED to MeshBasicMaterial
    map: THREE.ImageUtils.loadTexture(upDown),
    transparent: true,
  });
  upCC = new THREE.Mesh(new THREE.PlaneGeometry(4, 4), img);
  upCC.overdraw = true;
  upCC.rotation.x = -Math.PI / 2;
  upCC.position.set(0, 1, 0.8);
  scene.add(upCC);

  //DOWN CLOCKWISE ARROW
  var img = new THREE.MeshBasicMaterial({
    //CHANGED to MeshBasicMaterial
    map: THREE.ImageUtils.loadTexture(upDown),
    transparent: true,
  });
  downC = new THREE.Mesh(new THREE.PlaneGeometry(4, 4), img);
  downC.overdraw = true;
  downC.rotation.x = -Math.PI / 2;
  downC.position.set(0, -1, 0.8);
  scene.add(downC);

  //DOWN COUNTER CLOCKWISE ARROW
  var img = new THREE.MeshBasicMaterial({
    //CHANGED to MeshBasicMaterial
    map: THREE.ImageUtils.loadTexture(frontImageCW),
    transparent: true,
  });
  downCC = new THREE.Mesh(new THREE.PlaneGeometry(4, 4), img);
  downCC.overdraw = true;
  downCC.rotation.x = -Math.PI / 2;
  downCC.rotation.z = -Math.PI / 2;
  downCC.position.set(0.8, -1, 0);
  scene.add(downCC);

  frontC.visible = false;
  frontCC.visible = false;
  backC.visible = false;
  backCC.visible = false;
  leftC.visible = false;
  leftCC.visible = false;
  rightC.visible = false;
  rightCC.visible = false;
  upC.visible = false;
  upCC.visible = false;
  downC.visible = false;
  downCC.visible = false;
}

export function SetFrontC(bool) {
  frontC.visible = bool;
}
export function SetFrontCC(bool) {
  frontCC.visible = bool;
}
export function SetBackC(bool) {
  backC.visible = bool;
}
export function SetBackCC(bool) {
  backCC.visible = bool;
}
export function SetLeftC(bool) {
  leftC.visible = bool;
}
export function SetLeftCC(bool) {
  leftCC.visible = bool;
}
export function SetRightC(bool) {
  rightC.visible = bool;
}
export function SetRightCC(bool) {
  rightCC.visible = bool;
}
export function SetUpC(bool) {
  upC.visible = bool;
}
export function SetUpCC(bool) {
  upCC.visible = bool;
}
export function SetDownC(bool) {
  downC.visible = bool;
}
export function SetDownCC(bool) {
  downCC.visible = bool;
}
