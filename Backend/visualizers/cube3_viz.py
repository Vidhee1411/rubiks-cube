# ----------------------------------------------------------------------
# Matplotlib Rubik's cube simulator
# Written by Jake Vanderplas
# Adapted from cube code written by David Hogg
#   https://github.com/davidwhogg/MagicCube

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import widgets
from environments.cube3 import Cube3, Cube3State

import argparse
import re


class Quaternion:
    """Quaternion Rotation:

    Class to aid in representing 3D rotations via quaternions.
    """

    @classmethod
    def from_v_theta(cls, v, theta):
        """
        Construct quaternions from unit vectors v and rotation angles theta

        Parameters
        ----------
        v : array_like
            array of vectors, last dimension 3. Vectors will be normalized.
        theta : array_like
            array of rotation angles in radians, shape = v.shape[:-1].

        Returns
        -------
        q : quaternion object
            quaternion representing the rotations
        """
        theta = np.asarray(theta)
        v = np.asarray(v)
        s = np.sin(0.5 * theta)
        c = np.cos(0.5 * theta)

        v = v * s / np.sqrt(np.sum(v * v, -1))
        x_shape = v.shape[:-1] + (4,)

        x = np.ones(x_shape).reshape(-1, 4)
        x[:, 0] = c.ravel()
        x[:, 1:] = v.reshape(-1, 3)
        x = x.reshape(x_shape)

        return cls(x)

    def __init__(self, x):
        self.x = np.asarray(x, dtype=float)

    def __repr__(self):
        return "Quaternion:\n" + self.x.__repr__()

    def __mul__(self, other):
        # multiplication of two quaternions.
        # we don't implement multiplication by a scalar
        sxr = self.x.reshape(self.x.shape[:-1] + (4, 1))
        oxr = other.x.reshape(other.x.shape[:-1] + (1, 4))

        prod = sxr * oxr
        return_shape = prod.shape[:-1]
        prod = prod.reshape((-1, 4, 4)).transpose((1, 2, 0))

        ret = np.array([(prod[0, 0] - prod[1, 1]
                         - prod[2, 2] - prod[3, 3]),
                        (prod[0, 1] + prod[1, 0]
                         + prod[2, 3] - prod[3, 2]),
                        (prod[0, 2] - prod[1, 3]
                         + prod[2, 0] + prod[3, 1]),
                        (prod[0, 3] + prod[1, 2]
                         - prod[2, 1] + prod[3, 0])],
                       dtype=np.float,
                       order='F').T
        return self.__class__(ret.reshape(return_shape))

    def as_v_theta(self):
        """Return the v, theta equivalent of the (normalized) quaternion"""
        x = self.x.reshape((-1, 4)).T

        # compute theta
        norm = np.sqrt((x ** 2).sum(0))
        theta = 2 * np.arccos(x[0] / norm)

        # compute the unit vector
        v = np.array(x[1:], order='F', copy=True)
        v /= np.sqrt(np.sum(v ** 2, 0))

        # reshape the results
        v = v.T.reshape(self.x.shape[:-1] + (3,))
        theta = theta.reshape(self.x.shape[:-1])

        return v, theta

    def as_rotation_matrix(self):
        """Return the rotation matrix of the (normalized) quaternion"""
        v, theta = self.as_v_theta()

        shape = theta.shape
        theta = theta.reshape(-1)
        v = v.reshape(-1, 3).T
        c = np.cos(theta)
        s = np.sin(theta)

        mat = np.array([[v[0] * v[0] * (1. - c) + c,
                         v[0] * v[1] * (1. - c) - v[2] * s,
                         v[0] * v[2] * (1. - c) + v[1] * s],
                        [v[1] * v[0] * (1. - c) + v[2] * s,
                         v[1] * v[1] * (1. - c) + c,
                         v[1] * v[2] * (1. - c) - v[0] * s],
                        [v[2] * v[0] * (1. - c) - v[1] * s,
                         v[2] * v[1] * (1. - c) + v[0] * s,
                         v[2] * v[2] * (1. - c) + c]],
                       order='F').T
        return mat.reshape(shape + (3, 3))

    def rotate(self, points):
        rot_mat = self.as_rotation_matrix()
        return np.dot(points, rot_mat.T)


def project_points(points, q, view, vertical):
    """Project points using a quaternion q and a view v

    Parameters
    ----------
    points : array_like
        array of last-dimension 3
    q : Quaternion
        quaternion representation of the rotation
    view : array_like
        length-3 vector giving the point of view
    vertical : array_like
        direction of y-axis for view.  An error will be raised if it
        is parallel to the view.

    Returns
    -------
    proj: array_like
        array of projected points: same shape as points.
    """
    if vertical is None:
        vertical = [0, 1, 0]
    points = np.asarray(points)
    view = np.asarray(view)

    xdir = np.cross(vertical, view).astype(float)

    if np.all(xdir == 0):
        raise ValueError("vertical is parallel to v")

    xdir /= np.sqrt(np.dot(xdir, xdir))

    # get the unit vector corresponing to vertical
    ydir = np.cross(view, xdir)
    ydir /= np.sqrt(np.dot(ydir, ydir))

    # normalize the viewer location: this is the z-axis
    v2 = np.dot(view, view)
    zdir = view / np.sqrt(v2)

    # rotate the points
    rot_mat = q.as_rotation_matrix()
    r_pts = np.dot(points, rot_mat.T)

    # project the points onto the view
    dpoint = r_pts - view
    dpoint_view = np.dot(dpoint, view).reshape(dpoint.shape[:-1] + (1,))
    dproj = -dpoint * v2 / dpoint_view

    trans = list(range(1, dproj.ndim)) + [0]
    return np.array([np.dot(dproj, xdir),
                     np.dot(dproj, ydir),
                     -np.dot(dpoint, zdir)]).transpose(trans)


class InteractiveCube(plt.Axes):
    # Define some attributes
    base_face = np.array([[1, 1, 1],
                          [1, -1, 1],
                          [-1, -1, 1],
                          [-1, 1, 1],
                          [1, 1, 1]], dtype=float)
    stickerwidth = 0.9
    stickermargin = 0.5 * (1. - stickerwidth)
    stickerthickness = 0.001
    (d1, d2, d3) = (1 - stickermargin,
                    1 - 2 * stickermargin,
                    1 + stickerthickness)
    base_sticker = np.array([[d1, d2, d3], [d2, d1, d3],
                             [-d2, d1, d3], [-d1, d2, d3],
                             [-d1, -d2, d3], [-d2, -d1, d3],
                             [d2, -d1, d3], [d1, -d2, d3],
                             [d1, d2, d3]], dtype=float)

    base_face_centroid = np.array([[0, 0, 1]])
    base_sticker_centroid = np.array([[0, 0, 1 + stickerthickness]])

    def __init__(self, n, state=None, subgoal_num=-1, view=(0, 0, 10),
                 fig=None, **kwargs):
        # Define rotation angles and axes for the six sides of the cube
        x, y, z = np.eye(3)
        self.rots = [Quaternion.from_v_theta(x, theta) for theta in (np.pi / 2, -np.pi / 2)]
        self.rots += [Quaternion.from_v_theta(y, theta) for theta in (np.pi / 2, -np.pi / 2, np.pi, 2 * np.pi)]

        rect = [0, 0.16, 1, 0.84]
        self._move_list = []
        self.env: Cube3 = Cube3(subgoal_num)
        self.subgoal: np.array = self.env.subgoal

        self.N = n
        self._prevStates = []

        if state is not None:
            self.state: Cube3State = Cube3State(np.array([int(x) for x in state.split(",")]))
        else:
            self.state: Cube3State = self.env.generate_goal_states(1)[0]

        self._view = view
        self._start_rot = Quaternion.from_v_theta((1, -1, 0),
                                                  -np.pi / 6)

        self._grey_stickers = []
        self._black_stickers = []

        if fig is None:
            fig = plt.gcf()

        # disable default key press events
        callbacks = fig.canvas.callbacks.callbacks
        del callbacks['key_press_event']

        # add some defaults, and draw axes
        kwargs.update(dict(aspect=kwargs.get('aspect', 'equal'),
                           xlim=kwargs.get('xlim', (-1.7, 1.5)),
                           ylim=kwargs.get('ylim', (-1.5, 1.7)),
                           frameon=kwargs.get('frameon', False),
                           xticks=kwargs.get('xticks', []),
                           yticks=kwargs.get('yticks', [])))
        super(InteractiveCube, self).__init__(fig, rect, **kwargs)
        self.xaxis.set_major_formatter(plt.NullFormatter())
        self.yaxis.set_major_formatter(plt.NullFormatter())

        self._start_xlim = kwargs['xlim']
        self._start_ylim = kwargs['ylim']

        # Define movement for up/down arrows or up/down mouse movement
        self._ax_UD = (1, 0, 0)
        self._step_UD = 0.01

        # Define movement for left/right arrows or left/right mouse movement
        self._ax_LR = (0, -1, 0)
        self._step_LR = 0.01

        self._ax_LR_alt = (0, 0, 1)

        # Internal state variable
        self._active = False  # true when mouse is over axes
        self._button1 = False  # true when button 1 is pressed
        self._button2 = False  # true when button 2 is pressed
        self._event_xy = None  # store xy position of mouse event
        self._tab = False  # tab key pressed
        self._digits = []  # digits
        self._mirror: bool = False
        self._rotate_cube: bool = False

        self._current_rot = self._start_rot  # current rotation state
        self._face_polys = None
        self._sticker_polys = None

        self.plastic_color = 'black'

        # WHITE:0 - U, YELLOW:1 - D, BLUE:2 - L, GREEN:3 - R, ORANGE: 4 - B, RED: 5 - F
        self.face_colors = ["w", "#ffcf00",
                            "#ff6f00", "#cf0000",
                            "#00008f", "#009f0f",
                            "gray", "none"]

        self._initialize_arrays()

        self._draw_cube()

        # connect some GUI events
        self.figure.canvas.mpl_connect('button_press_event',
                                       self._mouse_press)
        self.figure.canvas.mpl_connect('button_release_event',
                                       self._mouse_release)
        self.figure.canvas.mpl_connect('motion_notify_event',
                                       self._mouse_motion)
        self.figure.canvas.mpl_connect('key_press_event',
                                       self._key_press)
        self.figure.canvas.mpl_connect('key_release_event',
                                       self._key_release)

        # self._initialize_widgets()

        # write some instructions
        """
        self.figure.text(0.05, 0.05,
                         "Mouse/arrow keys adjust view\n"
                         "U/D/L/R/B/F keys turn faces\n"
                         "(hold shift for counter-clockwise)",
                         size=10)
        """

    def set_rot(self, rot: int):
        if rot == 0:
            self._current_rot = Quaternion.from_v_theta((-0.53180525,  0.83020462,  0.16716299), 0.95063829)
        elif rot == 1:
            self._current_rot = Quaternion.from_v_theta((0.9248325,  0.14011997, -0.35362584), 2.49351394)

        self._draw_cube()

    def _initialize_arrays(self):
        # initialize centroids, faces, and stickers.  We start with a
        # base for each one, and then translate & rotate them into position.

        # Define N^2 translations for each face of the cube
        cubie_width = 2. / self.N
        translations = np.array([[[-1 + (i + 0.5) * cubie_width,
                                   -1 + (j + 0.5) * cubie_width, 0]]
                                 for i in range(self.N)
                                 for j in range(self.N)])

        # Create arrays for centroids, faces, stickers
        face_centroids = []
        faces = []
        sticker_centroids = []
        stickers = []
        colors = []

        factor = np.array([1. / self.N, 1. / self.N, 1])

        for i in range(6):
            rot_mat = self.rots[i].as_rotation_matrix()
            faces_t = np.dot(factor * self.base_face
                             + translations, rot_mat.T)
            stickers_t = np.dot(factor * self.base_sticker
                                + translations, rot_mat.T)
            face_centroids_t = np.dot(self.base_face_centroid
                                      + translations, rot_mat.T)
            sticker_centroids_t = np.dot(self.base_sticker_centroid
                                         + translations, rot_mat.T)
            # colors_i = i + np.zeros(face_centroids_t.shape[0], dtype=int)
            colors_i = np.arange(i * face_centroids_t.shape[0], (i + 1) * face_centroids_t.shape[0])

            # append face ID to the face centroids for lex-sorting
            face_centroids_t = np.hstack([face_centroids_t.reshape(-1, 3),
                                          colors_i[:, None]])
            sticker_centroids_t = sticker_centroids_t.reshape((-1, 3))

            faces.append(faces_t)
            face_centroids.append(face_centroids_t)
            stickers.append(stickers_t)
            sticker_centroids.append(sticker_centroids_t)

            colors.append(colors_i)

        self._face_centroids = np.vstack(face_centroids)
        self._faces = np.vstack(faces)
        self._sticker_centroids = np.vstack(sticker_centroids)
        self._stickers = np.vstack(stickers)

    def reset(self):
        self.state: Cube3State = self.env.generate_goal_states(1)[0]

    def _initialize_widgets(self):
        self._ax_reset = self.figure.add_axes([0.75, 0.05, 0.2, 0.075])
        self._btn_reset = widgets.Button(self._ax_reset, 'Reset View')
        self._btn_reset.on_clicked(self._reset_view)

        self._ax_solve = self.figure.add_axes([0.55, 0.05, 0.2, 0.075])
        self._btn_solve = widgets.Button(self._ax_solve, 'Solve Cube')
        self._btn_solve.on_clicked(self._solve_cube)

    def _project(self, pts):
        return project_points(pts, self._current_rot, self._view, [0, 1, 0])

    def _draw_cube(self):
        stickers = self._project(self._stickers)[:, :, :2]
        faces = self._project(self._faces)[:, :, :2]
        face_centroids = self._project(self._face_centroids[:, :3])
        sticker_centroids = self._project(self._sticker_centroids[:, :3])

        plastic_color = self.plastic_color
        # self._colors[np.ravel_multi_index((0,1,2),(6,N,N))] = 10
        colors_6 = self.state.colors // (self.N ** 2)
        # colors_6[10] = 3
        # colors_6[21] = 1
        # colors_6[16] = 5
        colors = np.asarray(self.face_colors)[colors_6]

        # ignore_idxs = np.array([i for i in range(len(self.state.colors)) if i not in self.subgoal])
        # if ignore_idxs.shape[0] > 0:
        #    argsort_idxs = np.argsort(self.state.colors)
        #    colors[argsort_idxs[ignore_idxs]] = "grey"
        for idx in self._grey_stickers:
            colors[idx] = "grey"
        for idx in self._black_stickers:
            colors[idx] = "k"

        face_zorders = -face_centroids[:, 2]
        sticker_zorders = -sticker_centroids[:, 2]

        if self._face_polys is None:
            # initial call: create polygon objects and add to axes
            self._face_polys = []
            self._sticker_polys = []

            for i in range(len(colors)):
                fp = plt.Polygon(faces[i], facecolor=plastic_color,
                                 zorder=face_zorders[i])
                sp = plt.Polygon(stickers[i], facecolor=colors[i],
                                 zorder=sticker_zorders[i])

                self._face_polys.append(fp)
                self._sticker_polys.append(sp)
                self.add_patch(fp)
                self.add_patch(sp)
        else:
            # subsequent call: update the polygon objects
            for i in range(len(colors)):
                self._face_polys[i].set_xy(faces[i])
                self._face_polys[i].set_zorder(face_zorders[i])
                self._face_polys[i].set_facecolor(plastic_color)

                self._sticker_polys[i].set_xy(stickers[i])
                self._sticker_polys[i].set_zorder(sticker_zorders[i])
                self._sticker_polys[i].set_facecolor(colors[i])

        self.figure.canvas.draw()

    def rotate(self, rot):
        self._current_rot = self._current_rot * rot

    def rotate_face(self, f, n=1, layer=0):
        self._move_list.append((f, n, layer))

        if not np.allclose(n, 0):
            move: str = "%s%s" % (f, n)
            move_idx: int = np.where(np.array(self.env.atomic_actions) == np.array(move))[0][0]
            self.state = self.env.next_state([self.state], [move_idx])[0][0]
            self._draw_cube()

    def move(self, move, draw=True):
        self.state = self.env.next_state([self.state], move)[0][0]
        if draw:
            self._draw_cube()

    def _reset_view(self):
        self.set_xlim(self._start_xlim)
        self.set_ylim(self._start_ylim)
        self._current_rot = self._start_rot
        self._draw_cube()

    def _solve_cube(self):
        move_list = self._move_list[:]
        for (face, n, layer) in move_list[::-1]:
            self.rotate_face(face, -n, layer)
        self._move_list = []

    def _key_press(self, event):
        if event.key == 'tab':
            self._tab = True
        elif event.key.upper() == "M":
            self._mirror = True
        elif event.key.upper() == "N":
            self._rotate_cube = True
        elif self._rotate_cube and (event.key.upper() in "ULF"):
            self.state = self.env.get_symmetry(self.state, "rotate_cube", event.key.upper())
        elif self._mirror and (event.key.upper() in "ULF"):
            self.state = self.env.get_symmetry(self.state, "mirror_cube", event.key.upper())
        elif event.key.isdigit():
            self._digits.append(event.key)
        elif event.key == 'right':
            if self._tab:
                ax_lr = self._ax_LR_alt
            else:
                ax_lr = self._ax_LR
            self.rotate(Quaternion.from_v_theta(ax_lr, 5 * self._step_LR))
        elif event.key == 'left':
            if self._tab:
                ax_lr = self._ax_LR_alt
            else:
                ax_lr = self._ax_LR
            self.rotate(Quaternion.from_v_theta(ax_lr, -5 * self._step_LR))
        elif event.key == 'up':
            self.rotate(Quaternion.from_v_theta(self._ax_UD, 5 * self._step_UD))
        elif event.key == 'down':
            self.rotate(Quaternion.from_v_theta(self._ax_UD, -5 * self._step_UD))
        elif event.key.upper() in 'LRUDBF':
            if self._tab:
                direction = -1
            else:
                direction = 1

            self.rotate_face(event.key.upper(), direction)

        elif event.key.upper() == 'P':
            self.figure.savefig('snapshot.jpg', transparent=True)

        elif event.key.upper() == 'S':
            self._solve_cube()

        elif event.key.upper() == 'K':
            idx = int("".join(self._digits))
            self._black_stickers = [idx]

            self._digits = []

        self._draw_cube()

        # is_solved: bool = self.env.is_solved([self.state])[0]
        # if is_solved:
        #    print("SOLVED!")
        #    self._move_list = []
        #    self._prevStates = []

    def _key_release(self, event):
        if event.key == 'tab':
            self._tab = False
        elif event.key.upper() == 'M':
            self._mirror = False
        elif event.key.upper() == 'N':
            self._rotate_cube = False

    def _mouse_press(self, event, event_x=None, event_y=None):
        if event_x is not None and event_y is not None:
            self._event_xy = (event_x, event_y)
            self._button1 = True
        else:
            self._event_xy = (event.x, event.y)
            if event.button == 1:
                self._button1 = True
            elif event.button == 3:
                self._button2 = True

    def _mouse_release(self, event):
        self._event_xy = None
        if event.button == 1:
            self._button1 = False
        elif event.button == 3:
            self._button2 = False

    def _mouse_motion(self, event, event_x=None, event_y=None):
        if self._button1 or self._button2:
            if event_x is not None and event_y is not None:
                dx = event_x - self._event_xy[0]
                dy = event_y - self._event_xy[1]
                self._event_xy = (event_x, event_y)
            else:
                dx = event.x - self._event_xy[0]
                dy = event.y - self._event_xy[1]
                self._event_xy = (event.x, event.y)

            if self._button1:
                if self._tab:
                    ax_lr = self._ax_LR_alt
                else:
                    ax_lr = self._ax_LR
                rot1 = Quaternion.from_v_theta(self._ax_UD,
                                               self._step_UD * dy)
                rot2 = Quaternion.from_v_theta(ax_lr,
                                               self._step_LR * dx)
                self.rotate(rot1 * rot2)

                self._draw_cube()

            if self._button2:
                factor = 1 - 0.003 * (dx + dy)
                xlim = self.get_xlim()
                ylim = self.get_ylim()
                self.set_xlim(factor * xlim[0], factor * xlim[1])
                self.set_ylim(factor * ylim[0], factor * ylim[1])

                self.figure.canvas.draw()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--init', type=str, default=None, help="")
    parser.add_argument('--moves', type=str, default=None, help="")
    parser.add_argument('--subgoal_num', type=int, default=-1, help="")
    args = parser.parse_args()

    state = None
    if args.init is not None:
        state = re.sub("[^0-9,]", "", args.init)

    cube_len = 3

    fig = plt.figure(figsize=(5, 5))

    interactive_cube = InteractiveCube(cube_len, state=state, subgoal_num=args.subgoal_num)
    fig.add_axes(interactive_cube)
    # interactive_cube.set_rot(1)

    plt.show()


if __name__ == '__main__':
    main()
