from dataclasses import dataclass

from agent.sim_agent import SimAgent
from sim_gym.constants import SPELLS
from sim_gym.state import State


class Action:
    def do(self, agent: SimAgent, state: State):
        pass

    def can_do(self, state: State):
        return True


@dataclass
class CastAction(Action):
    spell: str

    def do(self, agent: SimAgent, state: State):
        return agent.cast(self.spell)

    def can_do(self, state: State):
        return state.can_cast(self.spell)


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

    def can_do(self, state: State):
        return state.gcd_remaining > 0

    def do(self, agent: SimAgent, state: State):
        return agent.wait(state.gcd_remaining)


cast_actions = [
    CastAction(spell)
    for spell in SPELLS
]
cast_actions2 = [
    CastAction("Pestilence"),
    CastAction("BloodStrike"),
    CastAction("PlagueStrike"),
    CastAction("IcyTouch"),
    CastAction("Obliterate"),
    CastAction("HowlingBlast"),
    CastAction("FrostStrike"),
]
wait_actions = [
    Wait50(),
    WaitGCD(),
]
wait_actions2 = [
    #Wait50(),
    WaitGCD(),
]

ACTION_SPACE = cast_actions + wait_actions
ACTION_SPACE2 = cast_actions2 + wait_actions2


##SPELLS = [
 #   "Pestilence",
 #   "BloodStrike",
 #   "PlagueStrike",
 #   "IcyTouch",
 #   "HornOfWinter",
 #   "Obliterate",
 #   "HowlingBlast",
 #   "FrostStrike",
  #  "EmpowerRuneWeapon",
 #   "RaiseDead",
 #   "UnbreakableArmor",
 #   "BloodTap",
 #   "BloodFury",
 #   "HyperspeedAcceleration",
#]