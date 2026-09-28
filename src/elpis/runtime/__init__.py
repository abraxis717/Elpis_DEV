"""The runtime composition: one composition root and one durable receipt history.

``Runtime`` (``composition``) wires the subsystems explicitly around one ECS
kernel history (``history``). There is exactly one runtime composition; the
beta's numbered runtime generations are retired.
"""
from .composition import Runtime, RuntimeConfig
from .history import HistoryError, ReceiptHistory, ReceiptRecord, RecordedReceipt

__all__ = ("HistoryError", "ReceiptHistory", "ReceiptRecord", "RecordedReceipt", "Runtime", "RuntimeConfig")
