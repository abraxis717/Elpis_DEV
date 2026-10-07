//! Stable continuity refusal codes. The numeric values are C ABI (include/elpis/continuity.h).

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum Code {
    Uninitialized = 1,
    Unanchored = 2,
    AlreadyAnchored = 3,
    StateMismatch = 4,
    Corrupt = 5,
    PublicationRefused = 6,
    PublicationUncertain = 7,
    Locked = 8,
    AuthorityMismatch = 9,
    EvolutionPending = 10,
    EvolutionNotPending = 11,
    Exhausted = 12,
    Invalid = 13,
    Path = 14,
    Open = 15,
    LegacyStorage = 16,
    /// An operating-system failure outside publication (directory creation, open, lock, read).
    Io = 17,
    /// Testing library only: a simulated process death at a named step.
    TestingProcessDeath = 255,
}

impl Code {
    pub const ALL: [Code; 18] = [
        Code::Uninitialized,
        Code::Unanchored,
        Code::AlreadyAnchored,
        Code::StateMismatch,
        Code::Corrupt,
        Code::PublicationRefused,
        Code::PublicationUncertain,
        Code::Locked,
        Code::AuthorityMismatch,
        Code::EvolutionPending,
        Code::EvolutionNotPending,
        Code::Exhausted,
        Code::Invalid,
        Code::Path,
        Code::Open,
        Code::LegacyStorage,
        Code::Io,
        Code::TestingProcessDeath,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Code::Uninitialized => "CONTINUITY_UNINITIALIZED",
            Code::Unanchored => "CONTINUITY_UNANCHORED",
            Code::AlreadyAnchored => "CONTINUITY_ALREADY_ANCHORED",
            Code::StateMismatch => "CONTINUITY_STATE_MISMATCH",
            Code::Corrupt => "CONTINUITY_CORRUPT",
            Code::PublicationRefused => "CONTINUITY_PUBLICATION_REFUSED",
            Code::PublicationUncertain => "CONTINUITY_PUBLICATION_UNCERTAIN",
            Code::Locked => "CONTINUITY_LOCKED",
            Code::AuthorityMismatch => "CONTINUITY_AUTHORITY_MISMATCH",
            Code::EvolutionPending => "CONTINUITY_EVOLUTION_PENDING",
            Code::EvolutionNotPending => "CONTINUITY_EVOLUTION_NOT_PENDING",
            Code::Exhausted => "CONTINUITY_EXHAUSTED",
            Code::Invalid => "CONTINUITY_INVALID",
            Code::Path => "CONTINUITY_PATH",
            Code::Open => "CONTINUITY_OPEN",
            Code::LegacyStorage => "CONTINUITY_LEGACY_STORAGE",
            Code::Io => "CONTINUITY_IO",
            Code::TestingProcessDeath => "CONTINUITY_TESTING_PROCESS_DEATH",
        }
    }

    pub fn from_i32(value: i32) -> Option<Code> {
        Code::ALL.iter().copied().find(|c| *c as i32 == value)
    }
}
