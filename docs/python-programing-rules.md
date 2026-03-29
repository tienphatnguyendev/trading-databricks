# Fluent Python Rules

> Rules for writing idiomatic Python, distilled from Fluent Python 2nd Edition by Luciano Ramalho. For AI coding agents working on Python projects.

## Python Data Model

- Implement `__repr__` on every class using `type(self).__name__` and `!r` on attributes; implement `__str__` only when human-readable output should differ.
- Never call dunder methods directly; use the corresponding built-ins (`len(x)`, `iter(x)`, `repr(x)`) — they are faster and provide additional services.
- Implement `__hash__` and `__eq__` together or not at all; ensure the attributes used in both are immutable. Objects that compare equal must have the same hash.
- Implement `__getitem__` and `__len__` to create a minimal sequence protocol — this enables indexing, slicing, iteration, `in`, `sorted`, and `reversed`.
- Return your own type from `__getitem__` when handling slices, not the underlying storage type.
- Return `NotImplemented` (the singleton, not `NotImplementedError`) from operator dunders when the other operand's type is unsupported — this lets Python try the reflected method.
- Always return new objects from infix operators (`__add__`, `__mul__`); never mutate operands. Only augmented assignment methods (`__iadd__`) on mutable types should modify `self`.
- Implement reverse methods (`__radd__`) alongside forward methods (`__add__`) to support mixed-type operations.

## Sequences & Collections

- Prefer list comprehensions over `map`/`filter` with `lambda`; reserve comprehensions for building lists, not for side effects.
- Use generator expressions when feeding another constructor or processing large sequences — they yield lazily and save memory.
- Use tuple unpacking with `*rest` to capture excess items instead of indexing with hard-coded positions.
- Prefer `tuple` for fixed-length heterogeneous records and immutable sequences; they use less memory and signal immutability.
- Never put mutable objects inside tuples you intend to hash or use as dict keys.
- Use `collections.deque` for FIFO queues; never use `list.pop(0)` which is O(n).
- `sorted()` returns a new list; `list.sort()` sorts in place and returns `None` — never assign the result of `.sort()`.

## Dictionaries & Sets

- Use `d.setdefault(key, []).append(value)` or `collections.defaultdict` instead of checking `if key not in d` before inserting.
- Subclass `collections.UserDict` instead of `dict` for custom mappings — `dict`'s C-level shortcuts bypass overridden Python methods.
- Use set operations (`&`, `|`, `-`, `^`) instead of membership-testing loops; they are more concise and faster.
- Use `{1, 2, 3}` literal syntax for non-empty sets and `set()` for the empty set — `{}` creates an empty dict.
- Use `d1 | d2` (Python 3.9+) for dict merging; `{**d1, **d2}` for older versions.
- Use `list(dict.fromkeys(items))` for order-preserving deduplication.

## Text & Bytes

- Always pass `encoding='utf-8'` explicitly to `open()`; never rely on the platform default encoding.
- Apply the "Unicode sandwich": decode bytes at I/O boundaries, work with `str` internally, encode on output.
- Normalize Unicode with `unicodedata.normalize('NFC', text)` before comparing or storing user-supplied text.
- Use `str.casefold()` instead of `str.lower()` for case-insensitive comparisons.

## Data Classes & Records

- Use `@dataclass` for mutable data holders; `typing.NamedTuple` for immutable records; `@dataclass(frozen=True)` for hashable immutable data classes.
- Use `field(default_factory=list)` for mutable defaults in `@dataclass` — bare mutable defaults raise `ValueError` by design.
- Prefer `typing.NamedTuple` class syntax over `collections.namedtuple` for type hints and custom methods.
- Keep data classes focused on holding data; add behavior through external functions or protocols.

## Variables, References & Mutability

- Variables are labels (references), not boxes — assignment never copies data.
- Use `is` only for `None` and sentinel comparisons; use `==` for value equality everywhere else.
- Never use mutable objects (`[]`, `{}`) as default parameter values; use `None` and create inside the function body.
- Make defensive copies of mutable arguments in `__init__`: `self.items = list(items)`.
- Use `copy.deepcopy()` for independent clones of nested mutable structures; `list()`, `[:]`, `copy.copy()` are shallow and share inner references.
- `del` deletes the reference, not the object — the object is garbage-collected only when no references remain.

## Functions & Callables

- Use `operator.itemgetter`, `operator.attrgetter` instead of trivial lambdas — they are more readable and faster.
- Use `functools.partial` to freeze arguments; prefer over lambdas for inspectability via `.func`, `.args`, `.keywords`.
- Implement `__call__` for callable objects needing mutable state across invocations; for simpler cases use closures.
- Use keyword-only parameters (after bare `*`) to prevent positional-argument mistakes: `def f(a, *, key=None)`.
- Replace single-method classes with plain functions when the object has no internal state.
- Use `callable()` to check whether an object is callable — Python has nine kinds of callable objects.

## Decorators & Closures

- Always apply `@functools.wraps(func)` in custom decorators to preserve `__name__`, `__doc__`, and metadata.
- Use `@functools.cache` for short-lived scripts; `@lru_cache(maxsize=N)` for long-running processes to bound memory.
- Use `@singledispatch` for type-based dispatch instead of `if/elif isinstance(...)` chains — it is extensible across modules.
- Use `nonlocal` to rebind immutable free variables in closures; without it, assignment creates a new local.
- Decorators run at import time, not at call time — be cautious with side effects in decorator bodies.
- Ensure all arguments to `@cache`/`@lru_cache` decorated functions are hashable.

## Type Hints

- Annotate parameters with abstract types (`Sequence`, `Mapping`, `Iterable`) and return values with concrete types (`list`, `dict`) — Postel's Law.
- Use `Optional[X]` or `X | None` only when the parameter genuinely accepts `None`; `Optional` does not make a parameter optional — the default value does.
- Use `typing.Protocol` for structural subtyping ("static duck typing") when you don't control the implementing classes.
- Use `TypeVar` with `bound=` for generic functions that require specific operations on their type parameter.
- Do not chase 100% type coverage; leave inherently dynamic code unannotated rather than adding misleading hints.

## Classes, Protocols & Inheritance

- Favor composition over inheritance; avoid inheriting from concrete classes.
- Never subclass built-in types (`dict`, `list`, `str`); use `UserDict`, `UserList`, `UserString` from `collections`.
- Use `@classmethod` for alternative constructors (e.g., `frombytes`), not `@staticmethod`.
- Name mixin classes with a `Mixin` suffix; ensure they carry no instance state. Place mixins before the main base in the MRO.
- Always call `super()` in overriding methods including `__init__` for cooperative multiple inheritance.
- Prefer single underscore `_x` over double underscore `__x` for non-public attributes — name mangling is for accidental clash prevention, not privacy.
- Use `__slots__` only when you have thousands of instances and memory matters; redeclare in each subclass.
- Start with public attributes; convert to `@property` only when you need validation or computed values.
- Prefer `__getattr__` over `__getattribute__` for dynamic attribute access — it is only called when normal lookup fails.

## Iterators, Generators & Control Flow

- Use generator functions (`yield`) for lazy sequences; never build a full list just to iterate.
- Never make an iterable its own iterator — keep `__iter__` and `__next__` on separate classes.
- Use `yield from sub_gen()` to delegate to subgenerators instead of a manual re-yielding loop.
- Leverage `itertools` (`chain`, `islice`, `groupby`, `takewhile`, `product`) before writing custom generator logic.
- Use `@contextlib.contextmanager` for simple context managers; always wrap `yield` in `try/finally`.
- Place only the guarded statement in `try`; move success-path code to the `else` clause.
- Use `for/else` for search-failure detection without flag variables — `else` runs only when the loop completes without `break`.

## Concurrency & Async

- Use threads for I/O-bound work and processes for CPU-bound work; the GIL prevents parallel Python bytecode execution across threads.
- Start with `concurrent.futures.ThreadPoolExecutor`/`ProcessPoolExecutor` rather than raw `threading`/`multiprocessing`.
- Never call blocking functions (`time.sleep`, file I/O) inside `async def` — use `await asyncio.sleep()` and `await asyncio.to_thread()`.
- Use `asyncio.Semaphore` to throttle concurrent async operations — unlike thread pools, coroutines launch instantly and can overwhelm servers.
- Use `asyncio.run(main())` as the single entry point; avoid manually managing the event loop.
- Use `asyncio.create_task()` for concurrent coroutines and `await` for sequential dependencies.

## Metaprogramming

- Use `__init_subclass__` or class decorators before reaching for metaclasses — they solve most class-customization problems with far less complexity.
- Implement `__set_name__` in descriptor classes to auto-detect the attribute name instead of requiring it as a constructor argument.
- Never implement custom metaclasses in application code; restrict them to libraries and frameworks.
- A class can only have one metaclass — prefer `__init_subclass__` and class decorators because they avoid this constraint.
