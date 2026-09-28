# Immutable protocol limits, bound into genesis authority.
MAX_INT = (1 << 63) - 1
MAX_FRAME_BYTES = 262144
MAX_PAYLOAD_BYTES = 65536
MAX_STRING_BYTES = 1024

SCHEDULER_PROTOCOL_V1 = "active-mailbox-fifo/entity-id-before-founded-activation.v1"
SCHEDULER_PROTOCOL_V2 = "active-mailbox-global-arrival/founding-index-activation.v2"
SUPPORTED_SCHEDULER_PROTOCOLS = frozenset({
    SCHEDULER_PROTOCOL_V1,
    SCHEDULER_PROTOCOL_V2,
})

# Historical default descriptor. Existing callers and sealed histories using
# genesis_descriptor_digest() without an explicit scheduler override remain v1.
PROTOCOL = {
    "revision": "ecs.m1a.integration.v3",
    "max_int": MAX_INT,
    "max_frame_bytes": MAX_FRAME_BYTES,
    "max_payload_bytes": MAX_PAYLOAD_BYTES,
    "max_string_bytes": MAX_STRING_BYTES,
    "scheduler": SCHEDULER_PROTOCOL_V1,
    "lifecycle": "founded-active-dormant-terminal.v1",
}
