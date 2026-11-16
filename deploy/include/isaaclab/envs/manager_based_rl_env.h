// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <eigen3/Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include "isaaclab/manager/observation_manager.h"
#include "isaaclab/manager/action_manager.h"
#include "isaaclab/assets/articulation/articulation.h"
#include "isaaclab/algorithms/algorithms.h"
#include <iostream>
#include "isaaclab/utils/utils.h"

namespace isaaclab
{

class ObservationManager;
class ActionManager;

// ManagerBasedRLEnv - minimal C++ mirror of the Python mjlab environment, used on the robot.
// It has no physics: the "world" is the real robot (or the MuJoCo bridge).
//   cfg      : parsed deploy.yaml (step_dt, joint_ids_map, stiffness, damping,
//              default_joint_pos, commands, actions, observations)
//   robot    : Articulation whose data is refreshed from LowState
//   observation_manager / action_manager : built from cfg["observations"] / cfg["actions"]
//   alg      : policy runner (OrtRunner = ONNX Runtime)
class ManagerBasedRLEnv
{
public:
    // Constructor
    // Inputs: cfg = deploy.yaml node; robot_ = articulation (unitree::BaseArticulation).
    // Output: none. Copies joint map / default pose / gains into robot->data and creates
    //         the action and observation managers.
    ManagerBasedRLEnv(YAML::Node cfg, std::shared_ptr<Articulation> robot_)
    :cfg(cfg), robot(std::move(robot_))
    {
        // Parse configuration
        this->step_dt = cfg["step_dt"].as<float>();
        robot->data.joint_ids_map = cfg["joint_ids_map"].as<std::vector<float>>();
        robot->data.joint_pos.resize(robot->data.joint_ids_map.size());
        robot->data.joint_vel.resize(robot->data.joint_ids_map.size());

        { // default joint positions
            auto default_joint_pos = cfg["default_joint_pos"].as<std::vector<float>>();
            robot->data.default_joint_pos = Eigen::VectorXf::Map(default_joint_pos.data(), default_joint_pos.size());
        }
        { // joint stiffness and damping
            robot->data.joint_stiffness = cfg["stiffness"].as<std::vector<float>>();
            robot->data.joint_damping = cfg["damping"].as<std::vector<float>>();
        }

        robot->update();

        // load managers
        action_manager = std::make_unique<ActionManager>(cfg["actions"], this);
        observation_manager = std::make_unique<ObservationManager>(cfg["observations"], this);
    }

    // reset(): zero the gait phase and episode counter, refresh sensors, clear last action
    // and fill observation history buffers with the current observation.
    void reset()
    {
        global_phase = 0;
        episode_length = 0;
        robot->update();
        action_manager->reset();
        observation_manager->reset();
    }

    // step(): ONE POLICY STEP = robot->update() (read sensors) -> observation_manager->compute()
    // ({group_name: vector<float>}) -> alg->act(obs) (ONNX) -> action_manager->process_action()
    // (raw action * scale + offset = target joint angles). Called at 1/step_dt Hz.
    void step()
    {
        episode_length += 1;
        robot->update();
        auto obs = observation_manager->compute();
        auto action = alg->act(obs);
        action_manager->process_action(action);
    }

    float step_dt;
    
    YAML::Node cfg;

    std::unique_ptr<ObservationManager> observation_manager;
    std::unique_ptr<ActionManager> action_manager;
    std::shared_ptr<Articulation> robot;
    std::unique_ptr<Algorithms> alg;
    long episode_length = 0;
    float global_phase = 0.0f;
};

};