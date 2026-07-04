from dataclasses import dataclass, field


@dataclass
class FieldValue:
    value: str
    confidence: float  # 0.0–1.0
    source: str        # "barcode" | "ocr" | "rule" | "llm" | "agreement" | "blank"


EXTRACTION_FIELDS = (
    "recipient_name",
    "street_address",
    "city",
    "state",
    "zip_code",
    "tracking_number",
    "carrier",
)


@dataclass
class ExtractionResult:
    label_id: str
    recipient_name: FieldValue
    street_address: FieldValue
    city: FieldValue
    state: FieldValue
    zip_code: FieldValue
    tracking_number: FieldValue
    carrier: FieldValue
    llm_called: bool = False
    conflicts: list[str] = field(default_factory=list)
    processing_ms: int = 0
    llm_mode: str = "none"
    ocr_rotations_tried: int = 4
    # Additive telemetry rendered as metadata.llm; existing keys unchanged.
    llm_telemetry: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "label_id": self.label_id,
            "extracted": {
                name: {
                    "value": fv.value,
                    "confidence": fv.confidence,
                    "source": fv.source,
                }
                for name, fv in (
                    ("recipient_name", self.recipient_name),
                    ("street_address", self.street_address),
                    ("city", self.city),
                    ("state", self.state),
                    ("zip_code", self.zip_code),
                    ("tracking_number", self.tracking_number),
                    ("carrier", self.carrier),
                )
            },
            "metadata": {
                "llm_called": self.llm_called,
                "conflicts": self.conflicts,
                "processing_ms": self.processing_ms,
                "llm_mode": self.llm_mode,
                "ocr_rotations_tried": self.ocr_rotations_tried,
                "llm": self.llm_telemetry,
            },
        }
