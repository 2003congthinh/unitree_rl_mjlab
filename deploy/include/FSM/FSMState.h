#pragma once

#include "Types.h"
#include "param.h"
#include "FSM/BaseState.h"
#include "isaaclab/devices/keyboard/keyboard.h"
#include "unitree_joystick_dsl.hpp"

// FSMState - BaseState + robot I/O shared by all concrete states.
//   static lowcmd   : motor command message published to the robot (q, dq, kp, kd, tau per motor)
//   static lowstate : latest robot state (IMU, motor q/dq, joystick) received over DDS
//   static keyboard : optional keyboard input (for keyboard_velocity_commands)
class FSMState : public BaseState
{
public:
    // Inputs: state id and name. Reads config["FSM"][name]["transitions"], a map
    //   { target_state_name: "joystick DSL expression" }, e.g. { FixStand: "LT + up.on_pressed" },
    // compiles each expression (unitree_joystick_dsl.hpp) into a check on lowstate->joystick,
    // and adds a global safety check: if LowState times out -> go to "Passive".
    FSMState(int state, std::string state_string)
    : BaseState(state, state_string) 
    {
        spdlog::info("Initializing State_{} ...", state_string);

        auto transitions = param::config["FSM"][state_string]["transitions"];

        if(transitions)
        {
            auto transition_map = transitions.as<std::map<std::string, std::string>>();

            for(auto it = transition_map.begin(); it != transition_map.end(); ++it)
            {
                std::string target_fsm = it->first;
                if(!FSMStringMap.right.count(target_fsm))
                {
                    spdlog::warn("FSM State_'{}' not found in FSMStringMap!", target_fsm);
                    continue;
                }

                int fsm_id = FSMStringMap.right.at(target_fsm);

                std::string condition = it->second;
                unitree::common::dsl::Parser p(condition);
                auto ast = p.Parse();
                auto func = unitree::common::dsl::Compile(*ast);
                registered_checks.emplace_back(
                    std::make_pair(
                        [func]()->bool{ return func(FSMState::lowstate->joystick); },
                        fsm_id
                    )
                );
            }
        }

        // register for all states
        registered_checks.emplace_back(
            std::make_pair(
                []()->bool{ return lowstate->isTimeout(); },
                FSMStringMap.right.at("Passive")
            )
        );
    }

    // Every tick: refresh joystick edge detection and keyboard key.
    void pre_run()
    {
        lowstate->update();
        if(keyboard) keyboard->update();
    }

    // Every tick: publish the motor command filled by run().
    void post_run()
    {
        lowcmd->unlockAndPublish();
    }

    static std::unique_ptr<LowCmd_t> lowcmd;
    static std::shared_ptr<LowState_t> lowstate;
    static std::shared_ptr<Keyboard> keyboard;
};