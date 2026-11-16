// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"

// State_RLBase - runs a trained velocity policy (ONNX) on the robot.
// Two loops run in parallel:
//   * policy thread (50 Hz = deploy.yaml step_dt): env->step() = read sensors ->
//     build observation -> ONNX inference -> store processed action (target joint angles)
//   * FSM thread (1 kHz): run() copies the latest target angles into lowcmd; the motor
//     drivers' PD loop (kp, kd from deploy.yaml) tracks them.
class State_RLBase : public FSMState
{
public:
    // Constructor (src/State_RLBase.cpp): loads <policy_dir>/params/deploy.yaml and
    // <policy_dir>/exported/policy.onnx.
    State_RLBase(int state_mode, std::string state_string);

    // enter(): set kp/kd from deploy.yaml stiffness/damping, then start the policy thread.
    void enter()
    {
        // set gain
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[i].dq() = 0;
            lowcmd->msg_.motor_cmd()[i].tau() = 0;
        }

        env->robot->update();
        // Start policy thread
        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // Initialize timing
            auto sleepTill = clock::now() + dt;
            env->reset();

            while (policy_thread_running)
            {
                env->step();

                // Sleep
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            }
        });
    }

    void run();

    // exit(): stop and join the policy thread.
    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    std::thread policy_thread;
    bool policy_thread_running = false;
};

REGISTER_FSM(State_RLBase)
