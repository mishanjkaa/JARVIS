from __future__ import annotations

# Maximum number of memory.recall steps a single agent plan may contain. Enforced by
# app.brain.planner.plan_validator.validate_plan alongside the existing overall plan
# step cap, so a plan cannot pad itself with excessive single-key memory reads.
MAX_MEMORY_READS = 3
