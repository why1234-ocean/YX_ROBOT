# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).


from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg


from legged_lab.envs.YX1.YX1_config import (
    YX1_FlatAgentCfg,
    YX1_FlatEnvCfg,
    YX1_RoughAgentCfg,
    YX1_RoughEnvCfg,
)


from legged_lab.envs.YX1_new.YX1_new_config import (
    YX_new_FlatAgentCfg,
    YX_new_FlatEnvCfg,
    YX_new_RoughAgentCfg,
    YX_new_RoughEnvCfg,
)



from legged_lab.utils.task_registry import task_registry


task_registry.register("YX1_flat", BaseEnv, YX1_FlatEnvCfg(), YX1_FlatAgentCfg())
task_registry.register("YX1_rough", BaseEnv, YX1_RoughEnvCfg(), YX1_RoughAgentCfg())


task_registry.register("YX_new_flat", BaseEnv, YX_new_FlatEnvCfg(), YX_new_FlatAgentCfg())
task_registry.register("YX_new_rough", BaseEnv, YX_new_RoughEnvCfg(), YX_new_RoughAgentCfg())

