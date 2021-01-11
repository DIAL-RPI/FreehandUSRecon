Each row in `Demo_pos.txt` contains 9 values, which give the positioning information captured by a position tracking device.

1. The first value indicate the port number (0) of the tracking device.
2. The second value shows if the tracking device is working properly (0: OK, 1: Error). These two values are just indicators and may not be useful in this algorithm.
3. The following 7 values are in two groups to indicate the position of the corresponding ultrasound frame at this time point.
    1. 3 values for translations along x, y and z, respectively.
    2. The rest 4 values form a quaternion with elements in the order of (x, y, z, w) for rotation.

In our code `tools.py`, there is a function of `params_to_mat44`, which coverts such a set of 9 values explained above to a 4x4 transformation matrix. We get
matrices M(i) and M(i+1) at two consecutive time points, and compute their relative transformation Mi' as in the article. Then we decompose Mi' into 6 DOF to represent the relative transformation between two time points, which will then be used as the training label for the transformation betwen these two frames.
