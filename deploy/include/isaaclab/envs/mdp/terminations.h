#pragma once

#include "isaaclab/envs/manager_based_rl_env.h"

namespace isaaclab
{
namespace mdp
{

// bad_orientation(env, limit_angle)
//   Intended: true if the base tilt angle acos(-g_z) exceeds limit_angle (rad) -> Passive.
//   NOTE: the check is currently commented out and the function always returns false,
//   so this safety transition is disabled.
inline bool bad_orientation(ManagerBasedRLEnv* env, float limit_angle = 1.0)
{
    auto & asset = env->robot;
    auto & data = asset->data.projected_gravity_b;
//    return std::fabs(std::acos(-data[2])) > limit_angle;
    return false;
}

} 
} 