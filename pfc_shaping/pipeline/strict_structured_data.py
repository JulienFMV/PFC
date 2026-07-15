"""Fail-closed parsers for governance-critical structured artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

import yaml
from yaml.nodes import MappingNode
from yaml.resolver import BaseResolver


class StrictStructuredDataError(ValueError):
    """Raised when structured bytes have parser-ambiguous semantics."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


MAX_STRUCTURED_DEPTH = 128
MAX_STRUCTURED_CONTAINERS = 10_000


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise StrictStructuredDataError("unhashable YAML mapping key") from exc
        if duplicate:
            raise StrictStructuredDataError(f"duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_strict_yaml(raw: str) -> object:
    """Parse safe YAML while rejecting duplicate and merge-colliding keys."""

    try:
        value = yaml.load(raw, Loader=_UniqueKeySafeLoader)
    except (RecursionError, yaml.YAMLError) as exc:
        raise StrictStructuredDataError("invalid YAML") from exc
    try:
        _validate_structured_value(value)
    except RecursionError as exc:
        raise StrictStructuredDataError("structured-data nesting is too deep") from exc
    return value


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise StrictStructuredDataError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_strict_json(raw: str) -> object:
    """Parse JSON while rejecting duplicate keys at every nesting level."""

    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, RecursionError) as exc:
        raise StrictStructuredDataError("invalid JSON") from exc
    try:
        _validate_structured_value(value)
    except RecursionError as exc:
        raise StrictStructuredDataError("structured-data nesting is too deep") from exc
    return value


def _reject_json_constant(value: str) -> object:
    raise StrictStructuredDataError(f"non-finite JSON number: {value}")


def _validate_structured_value(
    value: object,
    *,
    active_containers: set[int] | None = None,
    validated_containers: dict[int, int] | None = None,
    depth: int = 0,
) -> int:
    if isinstance(value, float) and not math.isfinite(value):
        raise StrictStructuredDataError("non-finite structured-data number")
    is_mapping = isinstance(value, Mapping)
    is_sequence = isinstance(value, (list, tuple, set, frozenset))
    if not is_mapping and not is_sequence:
        return 0
    if depth >= MAX_STRUCTURED_DEPTH:
        raise StrictStructuredDataError("structured-data nesting is too deep")
    active = active_containers if active_containers is not None else set()
    validated = validated_containers if validated_containers is not None else {}
    identity = id(value)
    if identity in validated:
        height = validated[identity]
        if depth + height > MAX_STRUCTURED_DEPTH:
            raise StrictStructuredDataError("structured-data nesting is too deep")
        return height
    if identity in active:
        raise StrictStructuredDataError("cyclic structured-data alias")
    if len(active) + len(validated) >= MAX_STRUCTURED_CONTAINERS:
        raise StrictStructuredDataError("structured-data container limit exceeded")
    active.add(identity)
    max_child_height = 0
    try:
        if is_mapping:
            for key, item in value.items():  # type: ignore[union-attr]
                max_child_height = max(
                    max_child_height,
                    _validate_structured_value(
                        key,
                        active_containers=active,
                        validated_containers=validated,
                        depth=depth + 1,
                    ),
                )
                max_child_height = max(
                    max_child_height,
                    _validate_structured_value(
                        item,
                        active_containers=active,
                        validated_containers=validated,
                        depth=depth + 1,
                    ),
                )
        else:
            for item in value:  # type: ignore[union-attr]
                max_child_height = max(
                    max_child_height,
                    _validate_structured_value(
                        item,
                        active_containers=active,
                        validated_containers=validated,
                        depth=depth + 1,
                    ),
                )
    finally:
        active.remove(identity)
    height = max_child_height + 1
    validated[identity] = height
    return height
