"""Fixed-size ECS_G cognition continuity summary over the runtime receipt history.

The durable K1 lineage is the ordered sequence of ``ecs_g`` receipts of kind
``cognition.anchor`` / ``cognition.turn``. ``CognitionContinuity`` is the
incremental fold of that sequence. It carries only:

  * ``records``  how many continuity receipts were folded (lifetime);
  * ``anchored`` whether a lineage exists;
  * ``tip``      the K1 retained-state digest the next turn must start from;
  * ``fault``    the first lineage violation, if any (bounded text).

Because the fold is fixed-size it can be carried across compaction inside the
compaction checkpoint's bounded extension: the lineage of retired receipts is
never re-read, and restart never materializes lifetime history.

The rules are exactly the ones the runtime applied to the full receipt list:
an anchor must be the first and only anchor; a turn must continue the current
tip; the first turn of an unanchored legacy lineage establishes it. A
violation is recorded, not raised, so a history with a broken lineage still
opens; every lineage-dependent operation then refuses with
``HISTORY_CONTINUITY_INVALID``. The summary is ECS_C bookkeeping about
receipts, not cognitive state, and grants no authority.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ("CONTINUITY_SCHEMA", "CONTINUITY_KINDS", "CognitionContinuity", "ContinuityError")

CONTINUITY_SCHEMA = "elpis.runtime.cognition-continuity.v1"
CONTINUITY_KINDS = ("cognition.anchor", "cognition.turn")
_MAX_FAULT = 160
_MAX_RECORDS = (1 << 63) - 1


class ContinuityError(ValueError):
    def __init__(self, code: str, detail: str = ""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


def _digest_text(value):
    # Same acceptance rule the runtime has always applied to lineage bindings.
    if type(value) is not str or len(value) != 64:
        return None
    try:
        bytes.fromhex(value)
    except ValueError:
        return None
    return value


@dataclass(frozen=True)
class CognitionContinuity:
    records: int = 0
    anchored: bool = False
    tip: str | None = None
    fault: str | None = None

    def __post_init__(self):
        if type(self.records) is not int or not 0 <= self.records <= _MAX_RECORDS:
            raise ContinuityError("CONTINUITY_SUMMARY_INVALID", "records")
        if type(self.anchored) is not bool:
            raise ContinuityError("CONTINUITY_SUMMARY_INVALID", "anchored")
        if self.tip is not None and _digest_text(self.tip) is None:
            raise ContinuityError("CONTINUITY_SUMMARY_INVALID", "tip")
        if self.fault is not None and (type(self.fault) is not str or not 0 < len(self.fault) <= _MAX_FAULT):
            raise ContinuityError("CONTINUITY_SUMMARY_INVALID", "fault")
        if self.anchored != (self.tip is not None):
            raise ContinuityError("CONTINUITY_SUMMARY_INVALID", "anchored/tip")
        if self.records == 0 and (self.anchored or self.fault is not None):
            raise ContinuityError("CONTINUITY_SUMMARY_INVALID", "empty lineage")

    def _faulted(self, detail: str) -> "CognitionContinuity":
        return CognitionContinuity(self.records + 1, self.anchored, self.tip, detail[:_MAX_FAULT])

    def fold(self, record) -> "CognitionContinuity":
        """Fold one receipt record; records outside the lineage are ignored."""
        if record.subsystem != "ecs_g" or record.kind not in CONTINUITY_KINDS:
            return self
        if self.fault is not None:
            return CognitionContinuity(self.records + 1, self.anchored, self.tip, self.fault)
        bindings = dict(record.bindings)
        if record.kind == "cognition.anchor":
            state = _digest_text(bindings.get("state"))
            if state is None or record.digest != state or bindings.get("mechanism") != "1":
                return self._faulted("malformed cognition.anchor receipt")
            if self.anchored or self.records != 0:
                return self._faulted("cognition.anchor must be the first and only anchor")
            return CognitionContinuity(self.records + 1, True, state, None)
        before = _digest_text(bindings.get("state_before"))
        after = _digest_text(bindings.get("state_after"))
        if before is None or after is None or bindings.get("mechanism") != "1":
            return self._faulted("malformed cognition.turn state identity")
        if self.tip is not None and before != self.tip:
            return self._faulted(
                "cognition.turn state_before does not continue the durable K1 lineage")
        return CognitionContinuity(self.records + 1, True, after, None)

    def require_tip(self) -> str | None:
        """The current lineage tip, ``None`` if unanchored; refuses a broken lineage."""
        if self.fault is not None:
            raise ContinuityError("HISTORY_CONTINUITY_INVALID", self.fault)
        return self.tip

    def to_dict(self) -> dict:
        return {"schema": CONTINUITY_SCHEMA, "records": self.records, "anchored": self.anchored,
                "tip": self.tip, "fault": self.fault}

    @classmethod
    def from_dict(cls, value) -> "CognitionContinuity":
        if type(value) is not dict or set(value) != {"schema", "records", "anchored", "tip", "fault"}:
            raise ContinuityError("CONTINUITY_SUMMARY_INVALID", "shape")
        if value["schema"] != CONTINUITY_SCHEMA:
            raise ContinuityError("CONTINUITY_SUMMARY_INVALID", "schema")
        return cls(value["records"], value["anchored"], value["tip"], value["fault"])
