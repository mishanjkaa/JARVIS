from __future__ import annotations

from typing import Protocol

from app.brain.vision.models import LoadedVisionImage, VisionCropObservation, VisionEvidence, VisionProviderStatus


class VisionProvider(Protocol):
    name: str
    model: str

    def status(self) -> VisionProviderStatus:
        ...

    def describe_image(self, image: LoadedVisionImage, *, detail_level: str) -> VisionEvidence:
        ...

    def extract_text(self, image: LoadedVisionImage, *, language_hint: str, max_characters: int) -> VisionEvidence:
        ...

    def find_visual_element(
        self,
        image: LoadedVisionImage,
        *,
        query: str,
        max_results: int,
        strict_localization: bool = False,
    ) -> VisionEvidence:
        ...

    def verify_visual_crop(self, image: LoadedVisionImage) -> VisionCropObservation:
        ...

    def observe_full_frame(self, image: LoadedVisionImage) -> VisionCropObservation:
        ...
