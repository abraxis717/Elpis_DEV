//! Stable RuntimeCore refusal codes. Continuity codes (1..=17, 255) pass through with their own values and
//! names; RuntimeCore's own codes start at 64. The numeric values are C ABI (include/elpis/runtime.h).

use elpis_continuity::Code;

/// RuntimeCore's own refusals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum Rt {
    /// A malformed argument (NULL pointer, bad descriptor, inconsistent buffer lengths).
    Invalid = 64,
    /// The runtime is not open.
    Closed = 65,
    /// The open runtime's K1 lineage is bound to another native state.
    SubstrateSwitch = 67,
    /// The native K1 state could not be identified (its digest refused).
    EcsState = 68,
    /// The native K1 transaction refused (the authoritative state is unchanged).
    EcsRefused = 69,
    /// The candidate's source state was replaced meanwhile; the commit installed nothing.
    EcsStale = 70,
    /// A managed turn is already open.
    TurnOpen = 71,
    /// No managed turn is open (or another substrate's).
    TurnNotOpen = 72,
    /// An evolution attempt is already reserved and not finalized in this runtime.
    EvolutionInFlight = 73,
    /// No evolution attempt is in flight.
    EvolutionNotInFlight = 74,
}

impl Rt {
    pub const ALL: [Rt; 10] = [
        Rt::Invalid,
        Rt::Closed,
        Rt::SubstrateSwitch,
        Rt::EcsState,
        Rt::EcsRefused,
        Rt::EcsStale,
        Rt::TurnOpen,
        Rt::TurnNotOpen,
        Rt::EvolutionInFlight,
        Rt::EvolutionNotInFlight,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Rt::Invalid => "RUNTIME_INVALID",
            Rt::Closed => "RUNTIME_CLOSED",
            Rt::SubstrateSwitch => "COGNITION_SUBSTRATE_SWITCH",
            Rt::EcsState => "ECS_STATE",
            Rt::EcsRefused => "ECS_REFUSED",
            Rt::EcsStale => "ECS_STALE",
            Rt::TurnOpen => "RUNTIME_TURN_OPEN",
            Rt::TurnNotOpen => "RUNTIME_TURN_NOT_OPEN",
            Rt::EvolutionInFlight => "RUNTIME_EVOLUTION_IN_FLIGHT",
            Rt::EvolutionNotInFlight => "RUNTIME_EVOLUTION_NOT_IN_FLIGHT",
        }
    }

    pub fn from_i32(value: i32) -> Option<Rt> {
        Rt::ALL.into_iter().find(|c| *c as i32 == value)
    }
}

/// Any RuntimeCore refusal: a continuity code (the store's own, or a fail-stop disposition) or a runtime code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Continuity(Code),
    Runtime(Rt),
}

impl Error {
    pub fn code(self) -> i32 {
        match self {
            Error::Continuity(c) => c as i32,
            Error::Runtime(r) => r as i32,
        }
    }
}

impl From<Code> for Error {
    fn from(c: Code) -> Self {
        Error::Continuity(c)
    }
}

impl From<Rt> for Error {
    fn from(r: Rt) -> Self {
        Error::Runtime(r)
    }
}
