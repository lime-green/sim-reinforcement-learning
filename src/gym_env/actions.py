from dataclasses import dataclass

from agent.sim_agent import SimAgent
from gym_env.constants import SPELLS
from gym_env.state import State


class Action:
    def do(self, agent: SimAgent, state: State):
        pass


@dataclass
class CastAction(Action):
    spell: str

    def do(self, agent: SimAgent, state: State):
        return agent.cast(self.spell)


class WaitDuration(Action):
    ACTION_NAME: str = "WAIT_DURATION"
    DURATION = None

    def do(self, agent: SimAgent, state: State):
        return agent.wait(self.DURATION)


class Wait50(WaitDuration):
    """Wait 50ms"""
    DURATION = 50


class WaitGCD(WaitDuration):
    """Wait until GCD is ready"""

    def do(self, agent: SimAgent, state: State):
        return agent.wait(state.gcd_remaining)


cast_actions = [
    CastAction(spell)
    for spell in SPELLS
]

wait_actions = [
    Wait50(),
    WaitGCD(),
]

ACTION_SPACE = cast_actions + wait_actions
