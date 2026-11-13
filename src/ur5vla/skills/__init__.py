"""Engineered skills: scripted manipulation that produces training data.

A skill is a state machine that turns a perceived scene into a sequence of Cartesian legs, and
streams those legs as the absolute joint targets the control loop already executes. It plugs in
through `policy/scripted.py` as an ordinary PolicyBackend, so a scripted episode is recorded,
safety-checked and converted by exactly the same code as a teleoperated one -- the dataset cannot
tell them apart, which is the point.
"""
