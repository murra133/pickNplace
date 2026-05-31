"""Convert a Rhino Plane to robot pose fields.

Inputs expected in Grasshopper:
    P          Rhino.Geometry.Plane to convert.
    scale      Optional Rhino-to-robot scale. Use 1000 for mm Rhino models.
    conjugate  Optional bool. Set True only if your receiver expects inverse rotation.
    align_z    Optional bool. Use True for Rhino planes/rectangles whose +Z
               normal should drive robot TCP +X.
    reference_frame
               Optional string. Use "base" for tooltip_position reference_frame.
    plane_frame
               Optional string. Frame the input Rhino plane is already drawn in.
               Defaults to reference_frame. Use "world" only for planes drawn
               in the URDF world/robot-visualizer frame.

Outputs:
    x, y, z
    qx, qy, qz, qw
    position
    orientation
    reference_frame_out
"""

import math

import Rhino.Geometry as rg


try:
    scale
except NameError:
    scale = 1000.0

try:
    conjugate
except NameError:
    conjugate = False

try:
    align_z
except NameError:
    align_z = True

try:
    reference_frame
except NameError:
    reference_frame = "base"

try:
    plane_frame
except NameError:
    plane_frame = None

if P is None:
    raise ValueError("Connect a Rhino Plane to input P.")

origin = P.Origin

origin_x = origin.X / scale
origin_y = origin.Y / scale
origin_z = origin.Z / scale

x_axis_in = rg.Vector3d(P.XAxis)
y_axis_in = rg.Vector3d(P.YAxis)

if not x_axis_in.Unitize():
    raise ValueError("Plane X axis is invalid.")
if not y_axis_in.Unitize():
    raise ValueError("Plane Y axis is invalid.")

z_axis_in = rg.Vector3d.CrossProduct(x_axis_in, y_axis_in)
if not z_axis_in.Unitize():
    raise ValueError("Plane axes are parallel or invalid.")

if align_z:
    # Robot/tool +X is the faceplate normal. Rhino planes normally use +Z as
    # their normal, so map robot +X to plane +Z. Plane +X controls the roll
    # around that normal by becoming robot +Y.
    x_axis = z_axis_in
    y_axis = x_axis_in
    z_axis = rg.Vector3d.CrossProduct(x_axis, y_axis)
else:
    x_axis = x_axis_in
    y_axis = y_axis_in
    z_axis = z_axis_in

if not z_axis.Unitize():
    raise ValueError("Mapped plane axes are invalid.")

# Recompute Y so the output matrix is exactly orthonormal and right-handed.
y_axis = rg.Vector3d.CrossProduct(z_axis, x_axis)
y_axis.Unitize()

reference_frame_out = str(reference_frame or "base").lower()
plane_frame_in = str(plane_frame or reference_frame_out).lower()

if plane_frame_in == reference_frame_out:
    x = origin_x
    y = origin_y
    z = origin_z
elif plane_frame_in == "world" and reference_frame_out == "base":
    # sbot.urdf has world->base_link as Rz(-pi/2), so convert world geometry
    # to API base with inverse Rz(+pi/2).
    def world_to_base_point(px, py, pz):
        return -py, px, pz

    def world_to_base_vector(v):
        return rg.Vector3d(-v.Y, v.X, v.Z)

    x, y, z = world_to_base_point(origin_x, origin_y, origin_z)
    x_axis = world_to_base_vector(x_axis)
    y_axis = world_to_base_vector(y_axis)
    z_axis = world_to_base_vector(z_axis)
elif plane_frame_in == "base" and reference_frame_out == "world":
    def base_to_world_point(px, py, pz):
        return py, -px, pz

    def base_to_world_vector(v):
        return rg.Vector3d(v.Y, -v.X, v.Z)

    x, y, z = base_to_world_point(origin_x, origin_y, origin_z)
    x_axis = base_to_world_vector(x_axis)
    y_axis = base_to_world_vector(y_axis)
    z_axis = base_to_world_vector(z_axis)
else:
    raise ValueError('plane_frame and reference_frame must be "base" or "world".')

# Rotation matrix with plane axes as columns.
m00 = x_axis.X
m10 = x_axis.Y
m20 = x_axis.Z

m01 = y_axis.X
m11 = y_axis.Y
m21 = y_axis.Z

m02 = z_axis.X
m12 = z_axis.Y
m22 = z_axis.Z

trace = m00 + m11 + m22

if trace > 0.0:
    s = math.sqrt(trace + 1.0) * 2.0
    qw = 0.25 * s
    qx = (m21 - m12) / s
    qy = (m02 - m20) / s
    qz = (m10 - m01) / s
elif m00 > m11 and m00 > m22:
    s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
    qw = (m21 - m12) / s
    qx = 0.25 * s
    qy = (m01 + m10) / s
    qz = (m02 + m20) / s
elif m11 > m22:
    s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
    qw = (m02 - m20) / s
    qx = (m01 + m10) / s
    qy = 0.25 * s
    qz = (m12 + m21) / s
else:
    s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
    qw = (m10 - m01) / s
    qx = (m02 + m20) / s
    qy = (m12 + m21) / s
    qz = 0.25 * s

mag = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
if mag == 0.0:
    raise ValueError("Quaternion magnitude is zero.")

qx /= mag
qy /= mag
qz /= mag
qw /= mag

if conjugate:
    qx = -qx
    qy = -qy
    qz = -qz

position = {
    "x": x,
    "y": y,
    "z": z,
}

orientation = {
    "x": qx,
    "y": qy,
    "z": qz,
    "w": qw,
}
