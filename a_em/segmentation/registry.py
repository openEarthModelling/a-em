"""Registry for segmentation backends."""
from a_em.segmentation.base import SegmentationBackend


class SegmentationRegistry:
    """Class-level registry for managing available segmentation backends.

    Backends are registered by their ``name`` class attribute and can be
    retrieved by name or auto-discovered via ``find_compatible()``.
    """

    _backends: dict[str, type[SegmentationBackend]] = {}

    @classmethod
    def register(cls, backend_class: type[SegmentationBackend]):
        """Register a segmentation backend class.

        Args:
            backend_class: Subclass of SegmentationBackend with a ``name`` attribute.

        Raises:
            TypeError: If backend_class is not a SegmentationBackend subclass.
        """
        if not issubclass(backend_class, SegmentationBackend):
            raise TypeError(f"{backend_class} is not a SegmentationBackend subclass")
        if backend_class.name == 'abstract':
            raise ValueError(
                f"{backend_class.__name__} must define a 'name' class attribute"
            )
        if (backend_class.name in cls._backends and
            cls._backends[backend_class.name] is not backend_class):
            raise ValueError(
                f"Backend '{backend_class.name}' is already registered"
            )
        cls._backends[backend_class.name] = backend_class

    @classmethod
    def get(cls, name: str) -> SegmentationBackend:
        """Instantiate and return a registered backend by name.

        Args:
            name: Backend identifier (matching the class ``name`` attribute).

        Returns:
            Instance of the requested backend.

        Raises:
            ValueError: If no backend with the given name is registered.
        """
        if name not in cls._backends:
            available = ', '.join(cls.list_backends())
            raise ValueError(f"Unknown segmentation backend: '{name}'. Available: [{available}]")
        return cls._backends[name]()

    @classmethod
    def find_compatible(cls, microscope_type: str, particle_type: str) -> str:
        """Find the first registered backend that supports the given combination.

        Iterates in registration order and returns the name of the first
        backend whose ``supports()`` method returns True.

        Args:
            microscope_type: Type of microscope (e.g., 'TEM', 'SEM').
            particle_type: Type of particle (e.g., 'soot', 'spherical').

        Returns:
            Name of the first compatible backend.

        Raises:
            ValueError: If no compatible backend is found.
        """
        for name, backend_cls in cls._backends.items():
            if backend_cls.supports(microscope_type, particle_type):
                return name
        raise ValueError(f"No segmentation backend supports {microscope_type}/{particle_type}")

    @classmethod
    def list_backends(cls) -> list[str]:
        """Return a list of names of all registered backends."""
        return list(cls._backends.keys())

    @classmethod
    def clear(cls):
        """Remove all registered backends. Useful for test isolation."""
        cls._backends.clear()
