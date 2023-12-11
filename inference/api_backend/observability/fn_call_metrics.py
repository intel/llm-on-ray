import inspect
import time
from contextlib import AbstractContextManager, contextmanager
from enum import Enum
from functools import wraps
from typing import (
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    TypeVar,
)

from opentelemetry import context, trace
from ray.util import metrics

tracer = trace.get_tracer(__name__)


def outer_product(outer: Sequence[int], inner: Sequence[int]):
    """Take the sorted outer product.
    1. Remove duplicates
    2. Sort in increasing order (for use in latency ranges)
    """
    return sorted(list({x * y for x in outer for y in inner}))


class LatencyRange(Enum):
    short: List[float] = outer_product(
        range(1, 30), [1, 10, 100, 1000]
    )  # min: 1ms, max: 30s
    medium: List[float] = [x * 10 for x in short]  # min: 10ms, max: 300s
    slow: List[float] = [x * 100 for x in short]  # min: 100ms, max: 3000s (50 min)


class ClockUnit(int, Enum):
    ms = 1000
    s = 1


class MsClock:
    """A clock that tracks intervals in milliseconds"""

    def __init__(self, unit: ClockUnit = ClockUnit.ms):
        self.reset()
        self.unit = unit

    def reset(self):
        self.start_time = time.perf_counter()

    def interval(self):
        return (time.perf_counter() - self.start_time) * self.unit

    def reset_interval(self):
        interval = self.interval()
        self.reset()
        return interval


T = TypeVar("T")


class InstrumentTokenAsyncGenerator:
    """This class instruments an asynchronous generator.

    It gathers 3 metrics:
    1. Time to first time
    2. Time between tokens
    3. Total completion time

    Usage:

    @InstrumentTokenAsyncGenerator("my_special_fn")
    async def to_instrument():
        yield ...
    """

    all_instrument_names: Set[str] = set()

    def __init__(self, generator_name: str, latency_range=LatencyRange.short):
        self.generator_name = f"aviary_{generator_name}"
        self.latency_range = latency_range

    def __call__(
        self, async_generator_fn: Callable[..., AsyncGenerator[T, None]]
    ) -> Callable[..., AsyncGenerator[T, None]]:
        assert (
            self.generator_name not in self.all_instrument_names
        ), "This generator name was already used elsewhere. Please specify another one."
        self.all_instrument_names.add(self.generator_name)

        self.token_latency_histogram = metrics.Histogram(
            f"{self.generator_name}_per_token_latency_ms",
            f"Generator metrics for {self.generator_name}",
            boundaries=self.latency_range.value,
        )

        self.first_token_latency_histogram = metrics.Histogram(
            f"{self.generator_name}_first_token_latency_ms",
            f"Generator metrics for {self.generator_name}",
            boundaries=self.latency_range.value,
        )
        self.total_latency_histogram = metrics.Histogram(
            f"{self.generator_name}_total_latency_ms",
            f"Generator metrics for {self.generator_name}",
            boundaries=self.latency_range.value,
        )

        async def new_gen(*args, **kwargs):
            interval_clock = MsClock()
            total_clock = MsClock()
            is_first_token = True
            try:
                with TracingAsyncIterator(
                    self.generator_name, async_generator_fn(*args, **kwargs)
                ) as tracing_iterator:
                    async for x in tracing_iterator:
                        if is_first_token:
                            tracing_iterator.span.add_event("first_token")

                            self.first_token_latency_histogram.observe(
                                total_clock.interval()
                            )
                            interval_clock.reset()
                            is_first_token = False
                        else:
                            self.token_latency_histogram.observe(
                                interval_clock.reset_interval()
                            )
                        yield x
            finally:
                self.total_latency_histogram.observe(total_clock.interval())

        return new_gen


class TracingAsyncIterator:
    span: trace.Span
    span_context: context.Context

    iterations: int = 0

    def __init__(self, name: str, wrapped: AsyncIterator):
        self.wrapped = wrapped

        self.name = name
        self.span_context_manager: AbstractContextManager = (
            tracer.start_as_current_span(self.name)
        )

    def __enter__(self):
        self.span = self.span_context_manager.__enter__()
        self.span_context = context.get_current()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.span_context_manager.__exit__(exc_type, exc_value, traceback)

    def __aiter__(self):
        return self

    async def __anext__(self):
        with tracer.start_as_current_span(
            f"{self.name} iteration",
            context=self.span_context,
            attributes={"iteration": self.iterations},
        ):
            try:
                return await self.wrapped.__anext__()
            except Exception:
                self.__exit__(None, None, None)
                raise
            finally:
                self.iterations += 1
