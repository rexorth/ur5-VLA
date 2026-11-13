"""Turning camera frames into robot-frame geometry: AprilTag detection (tags.py), the camera's
pose on the arm (handeye.py) and the cube's pose on the table (cube.py).

Everything here needs OpenCV and is imported lazily by the callers, so the control loop, the
recorder and the converter still run in an environment without it.
"""
