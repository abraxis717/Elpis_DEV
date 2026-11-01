//! Stable RuntimeCore refusal codes: the authority codes (2..=5, 9..=13, 15) and RuntimeCore's own, from 64. The
//! numeric values are C ABI (include/elpis/runtime.h).
//!
//! ABI v8 retired every code of the persistent runtime and never reuses its values: the durable authority's storage
//! codes (1 UNINITIALIZED, 6 PUBLICATION_REFUSED, 7 PUBLICATION_UNCERTAIN, 8 LOCKED, 14 PATH, 16 LEGACY_STORAGE, 17 IO,
//! 255 TESTING_PROCESS_DEATH) and the checkpoint and durability-lane codes (76..=83). `elpis_runtime_code_name` names
//! none of them.

/// The authority refusal codes: the code space RuntimeCore has reported for its authority since ABI v1, kept with
/// their values and names (`CONTINUITY_*`); a session refusal with an equivalent maps to one (`authority::map`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum Code {
    Unanchored = 2,
    AlreadyAnchored = 3,
    StateMismatch = 4,
    Corrupt = 5,
    AuthorityMismatch = 9,
    EvolutionPending = 10,
    EvolutionNotPending = 11,
    Exhausted = 12,
    Invalid = 13,
    Open = 15,
}

impl Code {
    pub const ALL: [Code; 10] = [
        Code::Unanchored,
        Code::AlreadyAnchored,
        Code::StateMismatch,
        Code::Corrupt,
        Code::AuthorityMismatch,
        Code::EvolutionPending,
        Code::EvolutionNotPending,
        Code::Exhausted,
        Code::Invalid,
        Code::Open,
    ];

    /// The stable name (`elpis_runtime_code_name`).
    pub fn name(self) -> &'static std::ffi::CStr {
        match self {
            Code::Unanchored => c"CONTINUITY_UNANCHORED",
            Code::AlreadyAnchored => c"CONTINUITY_ALREADY_ANCHORED",
            Code::StateMismatch => c"CONTINUITY_STATE_MISMATCH",
            Code::Corrupt => c"CONTINUITY_CORRUPT",
            Code::AuthorityMismatch => c"CONTINUITY_AUTHORITY_MISMATCH",
            Code::EvolutionPending => c"CONTINUITY_EVOLUTION_PENDING",
            Code::EvolutionNotPending => c"CONTINUITY_EVOLUTION_NOT_PENDING",
            Code::Exhausted => c"CONTINUITY_EXHAUSTED",
            Code::Invalid => c"CONTINUITY_INVALID",
            Code::Open => c"CONTINUITY_OPEN",
        }
    }

    pub fn from_i32(value: i32) -> Option<Code> {
        Code::ALL.iter().copied().find(|c| *c as i32 == value)
    }
}

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
    /// The operation is outside its cognitive fuel budget (or the budget outside the ceiling); refused before
    /// any reserve, transaction or K1 mutation.
    Fuel = 75,
    /// The FMS-resident state is not resident (WARM) where a HOT operation requires it: refused before anything
    /// touches it, without promoting it or waiting; nothing bound, no fail-stop. Materialize it explicitly (a WARM
    /// operation) and retry.
    NotResident = 84,
    /// The session refused in a way the authority code space has no equivalent for (the exact session code is
    /// reported by `elpis_runtime_authority_status`): an operator event the transition law refuses, or a session
    /// that could not draw a nonce at open.
    Authority = 85,
    /// A typed observation names no present ACTIVE source of the session, or exceeds its source's per-turn bound.
    /// Refused before the native observe: the candidate is unchanged.
    ObservationRefused = 86,
    /// ABI v8: the session's memory could not be reserved at open (nothing was opened; the runtime stays closed).
    NoMem = 87,
    /// ABI v8: an operation of the retired persistent runtime (a legacy import or a reconciliation event): refused,
    /// nothing changed.
    Retired = 88,
}

impl Rt {
    pub const ALL: [Rt; 16] = [
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
        Rt::Fuel,
        Rt::NotResident,
        Rt::Authority,
        Rt::ObservationRefused,
        Rt::NoMem,
        Rt::Retired,
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
            Rt::Fuel => "COGNITION_FUEL_EXCEEDED",
            Rt::NotResident => "RUNTIME_NOT_RESIDENT",
            Rt::Authority => "RUNTIME_AUTHORITY",
            Rt::ObservationRefused => "RUNTIME_OBSERVATION_REFUSED",
            Rt::NoMem => "RUNTIME_NOMEM",
            Rt::Retired => "RUNTIME_RETIRED",
        }
    }

    pub fn from_i32(value: i32) -> Option<Rt> {
        Rt::ALL.into_iter().find(|c| *c as i32 == value)
    }
}

/// Any RuntimeCore refusal: an authority code (the session's, or a fail-stop disposition) or a runtime code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Authority(Code),
    Runtime(Rt),
}

impl Error {
    pub fn code(self) -> i32 {
        match self {
            Error::Authority(c) => c as i32,
            Error::Runtime(r) => r as i32,
        }
    }
}

impl From<Code> for Error {
    fn from(c: Code) -> Self {
        Error::Authority(c)
    }
}

impl From<Rt> for Error {
    fn from(r: Rt) -> Self {
        Error::Runtime(r)
    }
}
