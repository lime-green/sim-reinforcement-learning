import json

import numpy as np
from gym.spaces import Dict, Discrete, Box, MultiBinary

from .constants import BUFFS, DEBUFFS, SPELLS, RUNE_TYPE_MAP
from .normalization import ObservationNormalizer

SPELL_SET = set(SPELLS)


class State:
    def __init__(self, raw_state, normalizer):
        self._raw_state = raw_state
        self._abilities_map = {
            ability["name"]: ability
            for ability in self._raw_state["abilities"]
            if ability["name"] in SPELL_SET
        }
        self._debuffs_map = {
            debuff["name"]: debuff for debuff in self._raw_state["debuffs"]
        }
        self._buffs_map = {buff["name"]: buff for buff in self._raw_state["buffs"]}
        self._normalizer = normalizer

    @classmethod
    def create_normalizer(cls):
        return ObservationNormalizer(
            # exclude all the discrete values
            exclude_keys=[
                "isExecute35",
                "runeTypes",
                "debuffsActive",
                "buffsActive",
                "gcdAvailable",
            ]
        )

    @property
    def gcd_remaining(self):
        return self._raw_state["gcdRemaining"]

    @property
    def dps(self):
        return self._raw_state["dps"]

    @property
    def is_done(self):
        return self._raw_state["isDone"]

    @property
    def damage(self):
        return self._raw_state["totalDamage"]

    @property
    def runic_power(self):
        return self._raw_state["runicPower"]

    @property
    def debuffs(self):
        return self._debuffs_map

    @property
    def ability_damage(self):
        return self._raw_state["abilityDamage"]

    @property
    def ability_dps(self):
        return self._raw_state["abilityDPS"]

    @property
    def melee_dps(self):
        return self._raw_state["meleeDPS"]

    @property
    def disease_dps(self):
        return self._raw_state["diseaseDPS"]

    @property
    def time_elapsed(self):
        return self._raw_state["currentTime"]

    def can_cast(self, spell):
        return self._abilities_map[spell]["canCast"]

    def get_observations(self):
        return self._normalizer(self._get_observations())

    def _get_observations(self):
        return {
            # Discrete
            "isExecute35": int(self._raw_state["isExecute35"]),
            "runeTypes": [RUNE_TYPE_MAP[rt] for rt in self._raw_state["runeTypes"]],
            "debuffsActive": [
                int(self._debuffs_map[debuff]["isActive"]) for debuff in DEBUFFS
            ],
            "buffsActive": [int(self._buffs_map[buff]["isActive"]) for buff in BUFFS],
            "gcdAvailable": int(self._raw_state["gcdAvailable"]),
            # Continuous
            "abilityCDs": [ability["cdRemaining"] for ability in self.abilities],
            "abilityGCDs": [ability["gcdCost"] for ability in self.abilities],
            "debuffDurations": [
                self._debuffs_map[debuff]["duration"] for debuff in DEBUFFS
            ],
            "buffDurations": [self._buffs_map[buff]["duration"] for buff in BUFFS],
            "gcdRemaining": self.gcd_remaining,
            "runeCDs": self._raw_state["runeCDs"],
            "runeGraces": self._raw_state["runeGraces"],
        }

    @staticmethod
    def get_observation_space():
        # If we minmax scaled, then low could be 0, high 1, but not sure it matters
        return Dict(
            {
                # Discrete
                "isExecute35": Discrete(2),
                "runeTypes": Box(low=0, high=3, shape=(6,), dtype=np.uint8),
                "debuffsActive": MultiBinary(len(DEBUFFS)),
                "buffsActive": MultiBinary(len(BUFFS)),
                "gcdAvailable": Discrete(2),
                # Continuous
                "abilityCDs": Box(
                    low=0, high=1000 * 60 * 10, shape=(len(SPELLS),), dtype=np.uint32
                ),
                "abilityGCDs": Box(
                    low=0, high=1500, shape=(len(SPELLS),), dtype=np.uint16
                ),
                "debuffDurations": Box(
                    low=0, high=1000 * 60 * 10, shape=(len(DEBUFFS),), dtype=np.uint32
                ),
                "buffDurations": Box(
                    low=0, high=1000 * 60 * 10, shape=(len(BUFFS),), dtype=np.uint32
                ),
                "gcdRemaining": Box(low=0, high=1500, shape=(1,), dtype=np.uint16),
                "runeCDs": Box(low=0, high=1000 * 10, shape=(6,), dtype=np.uint16),
                "runeGraces": Box(low=0, high=2500, shape=(6,), dtype=np.uint16),
            }
        )

    @property
    def abilities(self):
        # deterministically order abilities
        abilities = [self._abilities_map[spell] for spell in SPELLS]

        return abilities

    def get_ability_mask(self):
        return [int(ability["canCast"]) for ability in self.abilities]

    def __repr__(self):
        return json.dumps(self._raw_state)
