"""Enumerations for Triage-Medley."""

from enum import Enum


class RETTSLevel(str, Enum):
    """RETTS triage colour levels, ordered by severity."""
    RED = "RED"
    ORANGE = "ORANGE"
    YELLOW = "YELLOW"
    GREEN = "GREEN"
    BLUE = "BLUE"

    @property
    def severity_rank(self) -> int:
        """Lower number = higher severity."""
        return {
            RETTSLevel.RED: 1,
            RETTSLevel.ORANGE: 2,
            RETTSLevel.YELLOW: 3,
            RETTSLevel.GREEN: 4,
            RETTSLevel.BLUE: 5,
        }[self]

    def __lt__(self, other: "RETTSLevel") -> bool:
        return self.severity_rank < other.severity_rank

    def __le__(self, other: "RETTSLevel") -> bool:
        return self.severity_rank <= other.severity_rank

    @classmethod
    def most_severe(cls, *levels: "RETTSLevel") -> "RETTSLevel":
        """Return the most severe (highest priority) level."""
        return min(levels, key=lambda l: l.severity_rank)


class QueuePriority(str, Enum):
    """Pre-triage queue priority levels."""
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    STANDARD = "STANDARD"


class Confidence(str, Enum):
    """Model confidence levels."""
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"


class ArrivalPathway(str, Enum):
    """Patient arrival pathway."""
    WALK_IN = "walk_in"
    REFERRAL_1177 = "referral_1177"
    AMBULANCE = "ambulance"


class ConsciousnessLevel(str, Enum):
    """AVPU consciousness scale."""
    ALERT = "Alert"
    VOICE = "Voice"
    PAIN = "Pain"
    UNRESPONSIVE = "Unresponsive"
