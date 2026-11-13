# Camera calibration

`wrist_handeye.yaml` is written by `ur5vla calibrate-handeye` and says where the wrist D415 sits
on the flange. It is **not** checked in by default: it belongs to one physical mounting of one
camera, and a stale copy is worse than none — every scripted grasp would miss by however far the
camera has moved since.

Re-run the calibration whenever the camera is unbolted, bumped, or re-seated:

    ur5vla calibrate-handeye -c configs/calibration.yaml

Tape an AprilTag 36h11 (36.2 mm black square, any id 0–15) flat on the table where the wrist
camera can see it from the `cube_look` pose, then let the arm orbit it. The command prints how
good the result is: under 3 mm of spread is good, over 8 mm will not grasp a 60 mm cube.
