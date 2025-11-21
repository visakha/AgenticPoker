"""
Functional programming utilities for the Agentic Poker engine.

This module provides pure utility functions for function composition,
piping, and working with immutable data structures.

Example:
    >>> from engine.functional_utils import pipe
    >>> result = pipe(
    ...     10,
    ...     lambda x: x * 2,
    ...     lambda x: x + 5,
    ...     lambda x: x ** 2
    ... )
    >>> result
    625
"""

from typing import TypeVar, Callable, Any, List
from functools import reduce

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


def pipe(value: T, *functions: Callable[[Any], Any]) -> Any:
    """
    Pipe a value through a sequence of functions.

    Args:
        value: Initial value
        *functions: Functions to apply in sequence

    Returns:
        Result of applying all functions

    Example:
        >>> pipe(5, lambda x: x * 2, lambda x: x + 3)
        13
    """
    return reduce(lambda acc, func: func(acc), functions, value)


def compose(*functions: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """
    Compose functions right-to-left.

    Args:
        *functions: Functions to compose

    Returns:
        Composed function

    Example:
        >>> add_five = lambda x: x + 5
        >>> double = lambda x: x * 2
        >>> f = compose(double, add_five)  # double(add_five(x))
        >>> f(10)
        30
    """

    def composed(value: Any) -> Any:
        return reduce(lambda acc, func: func(acc), reversed(functions), value)

    return composed


def curry(func: Callable[..., T]) -> Callable[..., Any]:
    """
    Curry a function (partial application).

    Args:
        func: Function to curry

    Returns:
        Curried function

    Example:
        >>> def add(a, b, c):
        ...     return a + b + c
        >>> curried_add = curry(add)
        >>> add_5 = curried_add(5)
        >>> add_5_10 = add_5(10)
        >>> add_5_10(3)
        18
    """
    from functools import wraps, partial

    @wraps(func)
    def curried(*args: Any, **kwargs: Any) -> Any:
        if len(args) + len(kwargs) >= func.__code__.co_argcount:
            return func(*args, **kwargs)
        return partial(curried, *args, **kwargs)

    return curried


def update_in(obj: T, path: List[str], value: Any) -> T:
    """
    Update a nested immutable structure (for dataclasses).

    Note: This is a helper for creating new instances with updated fields.
    Works with dataclasses by using `dataclasses.replace`.

    Args:
        obj: Dataclass instance
        path: List of attribute names to traverse
        value: New value to set

    Returns:
        New instance with updated value

    Example:
        >>> from dataclasses import dataclass, replace
        >>> @dataclass(frozen=True)
        ... class Point:
        ...     x: int
        ...     y: int
        >>> p = Point(1, 2)
        >>> update_in(p, ['x'], 10)
        Point(x=10, y=2)
    """
    from dataclasses import replace

    if len(path) == 1:
        return replace(obj, **{path[0]: value})

    # Nested update (recursive)
    current_attr = path[0]
    nested_obj = getattr(obj, current_attr)
    updated_nested = update_in(nested_obj, path[1:], value)
    return replace(obj, **{current_attr: updated_nested})


def filter_map(func: Callable[[T], U | None], items: List[T]) -> List[U]:
    """
    Map and filter in one pass (like Rust's filter_map).

    Args:
        func: Function that returns value or None
        items: List to process

    Returns:
        List of non-None results

    Example:
        >>> def parse_int(s):
        ...     try:
        ...         return int(s)
        ...     except ValueError:
        ...         return None
        >>> filter_map(parse_int, ['1', 'foo', '3', 'bar'])
        [1, 3]
    """
    result = []
    for item in items:
        value = func(item)
        if value is not None:
            result.append(value)
    return result


def partition(predicate: Callable[[T], bool], items: List[T]) -> tuple[List[T], List[T]]:
    """
    Partition a list into two lists based on a predicate.

    Args:
        predicate: Function to test each item
        items: List to partition

    Returns:
        Tuple of (items that pass, items that fail)

    Example:
        >>> is_even = lambda x: x % 2 == 0
        >>> partition(is_even, [1, 2, 3, 4, 5])
        ([2, 4], [1, 3, 5])
    """
    true_items = []
    false_items = []
    for item in items:
        if predicate(item):
            true_items.append(item)
        else:
            false_items.append(item)
    return true_items, false_items
