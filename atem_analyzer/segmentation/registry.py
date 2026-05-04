"""Registry for segmentation backends."""
from atem_analyzer.segmentation.base import SegmentationBackend


class SegmentationRegistry:
    _backends: dict[str, type[SegmentationBackend]] = {}

    @classmethod
    def register(cls, backend_class: type[SegmentationBackend]):
        if not issubclass(backend_class, SegmentationBackend):
            raise TypeError(f"{backend_class} is not a SegmentationBackend subclass")
        cls._backends[backend_class.name] = backend_class

    @classmethod
    def get(cls, name: str) -> SegmentationBackend:
        if name not in cls._backends:
            available = ', '.join(cls.list_backends())
            raise ValueError(f"Unknown segmentation backend: '{name}'. Available: [{available}]")
        return cls._backends[name]()

    @classmethod
    def find_compatible(cls, microscope_type: str, particle_type: str) -> str:
        for name, backend_cls in cls._backends.items():
            if backend_cls.supports(microscope_type, particle_type):
                return name
        raise ValueError(f"No segmentation backend supports {microscope_type}/{particle_type}")

    @classmethod
    def list_backends(cls) -> list[str]:
        return list(cls._backends.keys())

    @classmethod
    def clear(cls):
        cls._backends.clear()
