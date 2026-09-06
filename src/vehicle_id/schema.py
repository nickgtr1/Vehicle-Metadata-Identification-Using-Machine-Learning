from dataclasses import dataclass, field
from typing import Optional


# Shared output schema — every model (localisation, attributes, damage, accessories, VLM) returns one of these.
@dataclass
class VehicleProfile:
    make: Optional[str] = None
    model: Optional[str] = None
    body_type: Optional[str] = None
    colour: Optional[str] = None
    damage: list = field(default_factory=list)
    accessories: list = field(default_factory=list)
    confidence: dict = field(default_factory=dict)
    status: dict = field(default_factory=dict)