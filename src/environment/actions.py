from dataclasses import dataclass

from agent.sim_agent import SimAgent
from environment.constants import SPELLS
from environment.state import State


class Action:
    def do(self, agent: SimAgent, state: State):
        pass

    def can_do(self, state: State):
        return True


@dataclass
class CastAction(Action):
    spell: str

    @property
    def name(self):
        return f"CAST_{self.spell}"

    def do(self, agent: SimAgent, state: State):
        return agent.cast(self.spell)

    def can_do(self, state: State):
        return state.can_cast(self.spell)


class WaitDuration(Action):
    DURATION = None
    name = "WAIT_DURATION"

    def do(self, agent: SimAgent, state: State):
        return agent.wait(self.DURATION)


class DoNothing(Action):
    def do(self, agent: SimAgent, state: State):
        return agent.do_nothing()


class WaitGCD(WaitDuration):
    """Wait until GCD is ready"""

    name = "WAIT_GCD"

    def can_do(self, state: State):
        return state.gcd_remaining > 0

    def do(self, agent: SimAgent, state: State):
        return agent.wait(state.gcd_remaining)


cast_actions = [CastAction(spell) for spell in SPELLS]

wait_actions = [
    DoNothing(),
]

ACTION_SPACE = cast_actions + wait_actions
