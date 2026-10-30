# Create3-Wall-Following-Control
Master Thesis: Experimental comparison of PID, FLC, SMC and Adaptive PID controllers for wall-following with iRobot Create 3

Project Description

This repository contains Python scripts developed as part of a master's thesis focused on the implementation and experimental comparison of different control algorithms for wall-following navigation using the iRobot Create 3 mobile robot.

The control system was developed in Python using the ROS 2 (Robot Operating System 2) framework.

The main objective is to evaluate the performance of different control strategies in maintaining a desired distance from a wall under various experimental conditions.
Control System Architecture

The control program is implemented as a Finite State Machine (FSM), where individual states define different robot behaviors.

The Wall Follow state incorporates the implemented control algorithms, which determine the robot's control actions based on sensor measurements and the desired reference distance from the wall.
Implemented Controllers

Four control algorithms were implemented and experimentally evaluated:

    PID – Proportional-Integral-Derivative Controller

    FLC – Fuzzy Logic Controller

    SMC – Sliding Mode Controller

    Adaptive PID – Adaptive Proportional-Integral-Derivative Controller

Experimental Evaluation

The performance of the implemented controllers was evaluated through four groups of experimental tests:

    Nominal Operating Conditions – Evaluation of controller performance under standard operating conditions.

    Geometric Complexity – Evaluation of controller behavior when navigating geometrically challenging environments, including obstacle avoidance.

    Robustness to Parameter Variations – Evaluation of controller performance under variations in system parameters.

    External Disturbance Rejection – Evaluation of controller robustness and its ability to maintain stable operation under externally applied disturbances.

Experimental Platform

    Mobile Robot: iRobot Create 3

    Programming Language: Python

    Software Framework: ROS 2

    Control Architecture: Finite State Machine (FSM)

    Application: Wall-following navigation and experimental controller evaluation

Repository Contents

The repository contains Python scripts used for:

    Implementation of the state-machine-based control system.

    Implementation of PID, FLC, SMC, and Adaptive PID controllers.

    Experimental testing of wall-following performance.

    Processing and analysis of collected experimental data.

    Generation of graphical and statistical results.

Additional Notes

All control algorithms were experimentally evaluated on a physical iRobot Create 3 mobile robot under laboratory conditions.

The source code is provided for academic and research purposes.
