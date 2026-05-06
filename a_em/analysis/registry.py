"""Registry for analysis engines."""
from a_em.analysis.base import AnalysisEngine


class AnalysisEngineRegistry:
    """Central registry for all available analysis engines.

    Engines are registered by their ``name`` class attribute and can be
    retrieved by name or auto-discovered via ``find_compatible()``.
    """

    _engines: dict[str, type[AnalysisEngine]] = {}

    @classmethod
    def register(cls, engine_class: type[AnalysisEngine]):
        """Register an analysis engine class.

        Args:
            engine_class: Subclass of AnalysisEngine with a ``name`` attribute.

        Raises:
            TypeError: If engine_class is not an AnalysisEngine subclass.
        """
        if not issubclass(engine_class, AnalysisEngine):
            raise TypeError(f"{engine_class} is not an AnalysisEngine subclass")
        if engine_class.name == 'abstract':
            raise ValueError(
                f"{engine_class.__name__} must define a 'name' class attribute"
            )
        if engine_class.name in cls._engines and cls._engines[engine_class.name] is not engine_class:
            raise ValueError(
                f"Engine '{engine_class.name}' is already registered"
            )
        cls._engines[engine_class.name] = engine_class

    @classmethod
    def get(cls, name: str) -> AnalysisEngine:
        """Instantiate and return a registered engine by name.

        Args:
            name: Engine identifier (matching the class ``name`` attribute).

        Returns:
            Instance of the requested engine.

        Raises:
            ValueError: If no engine with the given name is registered.
        """
        if name not in cls._engines:
            available = ', '.join(cls.list_engines())
            raise ValueError(
                f"Unknown analysis engine: '{name}'. "
                f"Available: [{available}]"
            )
        return cls._engines[name]()

    @classmethod
    def find_compatible(cls, particle_type: str) -> str:
        """Find the first registered engine that supports the given particle type.

        Iterates in registration order and returns the name of the first
        engine whose ``supports()`` method returns True.

        Args:
            particle_type: e.g. 'soot', 'spherical'.

        Returns:
            Name of the first compatible engine.

        Raises:
            ValueError: If no compatible engine is found.
        """
        for name, engine_cls in cls._engines.items():
            if engine_cls.supports(particle_type):
                return name
        raise ValueError(f"No analysis engine for particle type: {particle_type}")

    @classmethod
    def list_engines(cls) -> list[str]:
        """Return a list of names of all registered engines."""
        return list(cls._engines.keys())

    @classmethod
    def clear(cls):
        """Remove all registered engines. Useful for test isolation."""
        cls._engines.clear()
