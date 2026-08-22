"""Software-only, two-tier local ECG service-loop prototype.

All results produced by this package are simulation evidence until Experiment
8B repeats the fixed scenarios on declared hardware.
"""

from .contracts import Acknowledgement, ActionRecord, ECGPacket, GatewayEvent

__all__ = ["Acknowledgement", "ActionRecord", "ECGPacket", "GatewayEvent"]
