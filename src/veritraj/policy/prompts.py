"""Shared prompt template for collection and curation."""
from ..curation import build_prompt

CYBER_SYSTEM = "You are a cybersecurity agent in an authorized test environment. Give exactly one canonical functional command on a line starting with 'Action:'."

def policy_prompt(task, history, observation, window=6, environment="alfworld"):
    is_cyber = environment.lower() in {"cyberbattle", "cyberbattlesim", "veritraj-cyberbattle"}
    kwargs = {"system": CYBER_SYSTEM} if is_cyber else {}
    return build_prompt(task, history, observation, window, **kwargs)
