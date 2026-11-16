// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <boost/bimap.hpp>
#include <string>
#include <any>
#include <utility>

// Global two-way map  state id (int) <-> state name (string), e.g. 1 <-> "Passive".
inline boost::bimap<int, std::string> FSMStringMap;

// BaseState - abstract life-cycle of one controller state:
//   enter()    called once when the FSM switches INTO this state
//   pre_run()  / run() / post_run()  called every 1 ms while active
//   exit()     called once when leaving
// registered_checks: list of (condition(), target_state_id); CtrlFSM::run_ polls them
// to decide transitions.
class BaseState
{
public:
    BaseState(int state, std::string state_string) : state_(state) 
    {
        FSMStringMap.insert({state, state_string});
    }

    virtual void enter() {}

    virtual void pre_run() {}
    virtual void run() {}
    virtual void post_run() {}

    virtual void exit() {}

    std::string getStateString() { return FSMStringMap.left.at(state_); }
    int getState() {return state_; }
    bool isState(int state) { return state_ == state; }
    std::vector<std::pair<std::function<bool()>, int>> registered_checks;
private:
    int state_;
};

// Factory registry: "State_Xxx" -> function(id, name) creating that state.
// REGISTER_FSM(State_Xxx) (placed after each state class) adds an entry at program
// start-up through a static registrar object, so CtrlFSM can build states by name.
using FsmFactory = std::function<std::shared_ptr<BaseState>(int, std::string)>;
using FsmMap     = std::unordered_map<std::string, FsmFactory>;

inline FsmMap& getFsmMap() {
    static FsmMap fsmMap;
    return fsmMap;
}

#define REGISTER_FSM(Derived) \
    inline std::shared_ptr<BaseState> __factory_##Derived(int s, std::string ss) {      \
        return std::make_shared<Derived>(s, ss);                                        \
    }                                                                                   \
    inline struct __registrar_##Derived {                                               \
        __registrar_##Derived() {                                                       \
            getFsmMap()[#Derived] = __factory_##Derived;                                \
        }                                                                               \
    } __registrar_instance_##Derived;                                                   \
    
