import { outlineMesh } from "./cube";
export var TWEEN = require("@tweenjs/tween.js");

// Matrix Multiplication for clockwise rotation - 90 degrees
export function matrixTransformationCW(a, b) {
  var l = 0,
    m = 1,
    n = -1,
    o = 0;
  var x = l * a + m * b;
  var y = n * a + o * b;
  return [x, y];
}

// Matrix Multiplication for anti-clockwise rotation = -90 degrees
export function matrixTransformationCCW(a, b) {
  var l = 0,
    m = -1,
    n = 1,
    o = 0;
  var x = l * a + m * b;
  var y = n * a + o * b;
  return [x, y];
}

export function updateXWithAnimation(pivot, ccw, duration) {
  var tween = new TWEEN.Tween(pivot.rotation)
    .to({ x: (ccw * Math.PI) / 2 }, duration) // relative animation
    .start();
  var tween = new TWEEN.Tween(outlineMesh.rotation)
    .to({ x: (ccw * Math.PI) / 2 }, duration) // relative animation
    .start();
}

export function updateYWithAnimation(pivot, ccw, duration) {
  var tween = new TWEEN.Tween(pivot.rotation)
    .to({ y: (ccw * Math.PI) / 2 }, duration) // relative animation
    .start();
  var tween = new TWEEN.Tween(outlineMesh.rotation)
    .to({ y: (ccw * -Math.PI) / 2 }, duration) // relative animation
    .start();
}

export function updateZWithAnimation(pivot, ccw, duration) {
  new TWEEN.Tween(pivot.rotation)
    .to({ z: (ccw * Math.PI) / 2 }, duration) // relative animation
    .start();
  new TWEEN.Tween(outlineMesh.rotation)
    .to({ z: (ccw * Math.PI) / 2 }, duration) // relative animation
    .start();
}
