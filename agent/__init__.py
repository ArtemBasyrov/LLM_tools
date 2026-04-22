"""
agent — orchestration package for rigorous, plan-following, self-critiquing
interaction with the local LLM.

Public entry points:

    from agent.orchestrator import Orchestrator
    from agent.plan import load_active, Plan, Step

The orchestrator wraps a single user turn and enforces:
  1. Triage of request complexity (agent.triage)
  2. Plan creation for complex tasks (tools.plan_tools + agent.plan)
  3. Verification of claimed step completion (agent.verifier)
  4. Self-critique of the final response (agent.critic)

All cross-context-window state lives on disk under ``LLM_PLAN_DIR``
(default ``~/.llm_plans``) so work can span sessions.
"""
