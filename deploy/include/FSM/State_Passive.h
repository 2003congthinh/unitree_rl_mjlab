// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"

// State_Passive - safe "damping only" mode (the start state and the fallback on errors).
// Config: config["FSM"]["Passive"] = { mode: [motor modes], kd: [damping per motor],
//                                      transitions: {...} }.
class State_Passive : public FSMState
{
public:
    // Constructor: copies the per-motor control mode from config into lowcmd.
    State_Passive(int state, std::string state_string = "Passive") 
    : FSMState(state, state_string) 
    {
        auto motor_mode = param::config["FSM"]["Passive"]["mode"];
        if(motor_mode.IsDefined())
        {
            auto values = motor_mode.as<std::vector<int>>();
            for(int i(0); i<values.size(); ++i)
            {
                lowcmd->msg_.motor_cmd()[i].mode() = values[i];
            }
        }
    } 

    // enter(): kp = 0, kd = config value, no feed-forward -> motors only resist motion.
    void enter()
    {
        // set gain
        static auto kd = param::config["FSM"]["Passive"]["kd"].as<std::vector<float>>();
        for(int i(0); i < kd.size(); ++i)
        {
            auto & motor = lowcmd->msg_.motor_cmd()[i];
            motor.kp() = 0;
            motor.kd() = kd[i];
            motor.dq() = 0;
            motor.tau() = 0;
        }
    }

    // run(): target q = current q, so switching to another state starts from here smoothly.
    void run()
    {
        for(int i(0); i < lowcmd->msg_.motor_cmd().size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].q() = lowstate->msg_.motor_state()[i].q();
        }
    }
};

REGISTER_FSM(State_Passive)
