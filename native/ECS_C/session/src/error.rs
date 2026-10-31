//! Stable ECS_C refusal codes. The numeric values are C ABI: 1..33 and 255 are the retiring durable authority's
//! (include/elpis/ecs_c_authority.h, unchanged); 34..37 are the volatile session's (include/elpis/ecs_c_session.h).
//! A session returns only the codes of its own law and [`Code::VOLATILE`]: never a storage code.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum Code {
    /// A malformed request: a bad argument, a reserved byte, an event of the wrong shape.
    Invalid = 1,
    /// Compare-and-publish: the request names another generation or state root than the current one, or a
    /// founding request names an entity id that the current state would not found.
    Stale = 2,
    /// The sender may not send this event (kind not authorized, sender not ACTIVE).
    Unauthorized = 3,
    UnknownEntity = 4,
    /// An illegal lifecycle transition, a transition from a state the entity is not in, or one that would leave
    /// the authority without an ACTIVE operator.
    Lifecycle = 5,
    /// The sender sequence is at or below the sender's watermark (a replayed or retried request that landed).
    Duplicate = 6,
    /// The sender sequence skips past watermark + 1.
    SequenceGap = 7,
    /// The provisioned entity capacity is exhausted. Refused before any write; nothing is discarded.
    Capacity = 8,
    /// A counter (generation, revision, sequence, count) would pass its bound. Refused before any write.
    Exhausted = 9,
    /// Evidence or a version change names another version than the source's current one.
    SourceStale = 10,
    ScopeConflict = 11,
    Unanchored = 12,
    AlreadyAnchored = 13,
    /// The cognition lineage is not at the stated identity, or not of the stated format.
    StateMismatch = 14,
    EvolutionMismatch = 15,
    EvolutionPending = 16,
    EvolutionNotPending = 17,
    LegacyRefused = 18,
    /// A write failed before the transition became durable: nothing changed.
    PublicationRefused = 19,
    /// The durable outcome is unknown (or the transition is durable but unwitnessed): the handle is closed and
    /// reopening resolves to exactly one complete authority.
    PublicationUncertain = 20,
    Locked = 21,
    Io = 22,
    Path = 23,
    /// The authority directory, the deployment key or a witness is not protected as the threat model requires.
    Unprotected = 24,
    NotProvisioned = 25,
    AlreadyProvisioned = 26,
    /// Provisioning never reached its publication point; provision again.
    ProvisionIncomplete = 27,
    Corrupt = 28,
    /// Recovery refused to open: the recovery report names the disposition.
    RecoveryRefused = 29,
    /// The authority opened in a restricted or reconcile-only mode: only the matching operator reconciliation
    /// is accepted.
    ReconcileRequired = 30,
    /// The reconciliation does not match what recovery observed, or there is nothing to reconcile.
    NotReconcilable = 31,
    Open = 32,
    Closed = 33,
    /// Volatile session: a coordinate names another session (or another instance of one).
    SessionMismatch = 34,
    /// Volatile session: the operation existed only for persistence (legacy import of a durable register, rollback,
    /// unauthenticated-slot or witness reconciliation) and is retired: a session has no storage to reconcile.
    PersistenceRetired = 35,
    /// Volatile session: a reservation of session memory failed at creation; nothing was created.
    NoMem = 36,
    /// Volatile session: no nonce was supplied and the kernel's random source refused one; nothing was created.
    EntropyUnavailable = 37,
    /// Testing library only: a simulated process death at a named step.
    TestingProcessDeath = 255,
}

impl Code {
    pub const ALL: [Code; 34] = [
        Code::Invalid,
        Code::Stale,
        Code::Unauthorized,
        Code::UnknownEntity,
        Code::Lifecycle,
        Code::Duplicate,
        Code::SequenceGap,
        Code::Capacity,
        Code::Exhausted,
        Code::SourceStale,
        Code::ScopeConflict,
        Code::Unanchored,
        Code::AlreadyAnchored,
        Code::StateMismatch,
        Code::EvolutionMismatch,
        Code::EvolutionPending,
        Code::EvolutionNotPending,
        Code::LegacyRefused,
        Code::PublicationRefused,
        Code::PublicationUncertain,
        Code::Locked,
        Code::Io,
        Code::Path,
        Code::Unprotected,
        Code::NotProvisioned,
        Code::AlreadyProvisioned,
        Code::ProvisionIncomplete,
        Code::Corrupt,
        Code::RecoveryRefused,
        Code::ReconcileRequired,
        Code::NotReconcilable,
        Code::Open,
        Code::Closed,
        Code::TestingProcessDeath,
    ];

    /// The volatile session's own codes (not in [`Code::ALL`], whose set is the durable C ABI's).
    pub const VOLATILE: [Code; 4] =
        [Code::SessionMismatch, Code::PersistenceRetired, Code::NoMem, Code::EntropyUnavailable];

    pub fn name(self) -> &'static str {
        match self {
            Code::Invalid => "ECS_C_INVALID",
            Code::Stale => "ECS_C_STALE",
            Code::Unauthorized => "ECS_C_UNAUTHORIZED",
            Code::UnknownEntity => "ECS_C_UNKNOWN_ENTITY",
            Code::Lifecycle => "ECS_C_LIFECYCLE",
            Code::Duplicate => "ECS_C_DUPLICATE",
            Code::SequenceGap => "ECS_C_SEQUENCE_GAP",
            Code::Capacity => "ECS_C_CAPACITY",
            Code::Exhausted => "ECS_C_EXHAUSTED",
            Code::SourceStale => "ECS_C_SOURCE_STALE",
            Code::ScopeConflict => "ECS_C_SCOPE_CONFLICT",
            Code::Unanchored => "ECS_C_UNANCHORED",
            Code::AlreadyAnchored => "ECS_C_ALREADY_ANCHORED",
            Code::StateMismatch => "ECS_C_STATE_MISMATCH",
            Code::EvolutionMismatch => "ECS_C_EVOLUTION_MISMATCH",
            Code::EvolutionPending => "ECS_C_EVOLUTION_PENDING",
            Code::EvolutionNotPending => "ECS_C_EVOLUTION_NOT_PENDING",
            Code::LegacyRefused => "ECS_C_LEGACY_REFUSED",
            Code::PublicationRefused => "ECS_C_PUBLICATION_REFUSED",
            Code::PublicationUncertain => "ECS_C_PUBLICATION_UNCERTAIN",
            Code::Locked => "ECS_C_LOCKED",
            Code::Io => "ECS_C_IO",
            Code::Path => "ECS_C_PATH",
            Code::Unprotected => "ECS_C_UNPROTECTED",
            Code::NotProvisioned => "ECS_C_NOT_PROVISIONED",
            Code::AlreadyProvisioned => "ECS_C_ALREADY_PROVISIONED",
            Code::ProvisionIncomplete => "ECS_C_PROVISION_INCOMPLETE",
            Code::Corrupt => "ECS_C_CORRUPT",
            Code::RecoveryRefused => "ECS_C_RECOVERY_REFUSED",
            Code::ReconcileRequired => "ECS_C_RECONCILE_REQUIRED",
            Code::NotReconcilable => "ECS_C_NOT_RECONCILABLE",
            Code::Open => "ECS_C_OPEN",
            Code::Closed => "ECS_C_CLOSED",
            Code::TestingProcessDeath => "ECS_C_TESTING_PROCESS_DEATH",
            Code::SessionMismatch => "ECS_C_SESSION_MISMATCH",
            Code::PersistenceRetired => "ECS_C_PERSISTENCE_RETIRED",
            Code::NoMem => "ECS_C_NOMEM",
            Code::EntropyUnavailable => "ECS_C_ENTROPY_UNAVAILABLE",
        }
    }

    pub fn from_i32(value: i32) -> Option<Code> {
        Code::ALL.iter().copied().find(|c| *c as i32 == value)
    }
}
