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
    /// The operation is outside its cognitive fuel budget (or the budget outside the ceiling); refused before
    /// any reserve, transaction or K1 mutation.
    Fuel = 75,
    /// The K1 checkpoint store is not provisioned as required (missing, wrong files or sizes, locked, refused).
    CheckpointInvalid = 76,
    /// A K1 checkpoint slot could not be written, synced, or read back verified (its bytes are not exactly the
    /// envelope written); the authorized slot is intact.
    CheckpointIo = 77,
    /// The K1 state's envelope is not the provisioned checkpoint capacity (another shape): refused, never resized.
    CheckpointShape = 78,
    /// A newer unauthorized checkpoint candidate needs explicit operator reconciliation (discard or adopt) first.
    CheckpointUnresolved = 79,
    /// A durable publication is pending or running; the operation would change the lineage under it.
    DurableBusy = 80,
    /// The durable publication waited for has not resolved yet (a wait that timed out). Not a failure.
    DurablePending = 81,
    /// A recoverable-state publication was requested without an attached checkpoint store: no owner could hold the
    /// complete state, so nothing was captured or published (identity-only publication is a separate request).
    RecoveryUnavailable = 82,
    /// An identity-only publication was requested while a checkpoint store is attached: every publication must then
    /// be recoverable (continuity never names an identity the attached store does not hold).
    CheckpointAttached = 83,
    /// The FMS-resident state is not resident (WARM) where a HOT operation requires it (demoted, or a move in
    /// flight): refused before anything touches it, without promoting it, reading the cold store or waiting; nothing
    /// bound, no fail-stop. Materialize it explicitly (a WARM operation) and retry.
    NotResident = 84,
    /// ABI v6, ECS_C authority only: the authority refused in a way continuity has no code for. At open: it is not
    /// provisioned, not protected, or recovery opened it restricted / reconcile-only or refused it (the operator
    /// reconciles it with the runtime closed). Otherwise: the authority does not authorize the operation. The exact
    /// ECS_C code is reported by `elpis_runtime_authority_status`.
    Authority = 85,
    /// ABI v6, ECS_C authority only: a typed observation names no present ACTIVE source of the authority, or exceeds
    /// its source's per-turn bound. Refused before the native observe: the candidate is unchanged.
    ObservationRefused = 86,
}

impl Rt {
    pub const ALL: [Rt; 22] = [
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
        Rt::CheckpointInvalid,
        Rt::CheckpointIo,
        Rt::CheckpointShape,
        Rt::CheckpointUnresolved,
        Rt::DurableBusy,
        Rt::DurablePending,
        Rt::RecoveryUnavailable,
        Rt::CheckpointAttached,
        Rt::NotResident,
        Rt::Authority,
        Rt::ObservationRefused,
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
            Rt::CheckpointInvalid => "CHECKPOINT_INVALID",
            Rt::CheckpointIo => "CHECKPOINT_IO",
            Rt::CheckpointShape => "CHECKPOINT_SHAPE",
            Rt::CheckpointUnresolved => "CHECKPOINT_UNRESOLVED",
            Rt::DurableBusy => "RUNTIME_DURABLE_BUSY",
            Rt::DurablePending => "RUNTIME_DURABLE_PENDING",
            Rt::RecoveryUnavailable => "RUNTIME_RECOVERY_UNAVAILABLE",
            Rt::CheckpointAttached => "RUNTIME_CHECKPOINT_ATTACHED",
            Rt::NotResident => "RUNTIME_NOT_RESIDENT",
            Rt::Authority => "RUNTIME_AUTHORITY",
            Rt::ObservationRefused => "RUNTIME_OBSERVATION_REFUSED",
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
