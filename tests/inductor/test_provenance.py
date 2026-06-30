# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import pytest


PROVENANCE_SCHEMA = {
    "type": "object",
    "required": ["kernels"],
    "properties": {
        "kernels": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": [
                    "kernel_name",
                    "compiled_kernel",
                    "handles",
                    "ir_stages",
                ],
                "properties": {
                    "kernel_name": {"type": "string", "minLength": 1},
                    "compiled_kernel": {"type": "string", "minLength": 1},
                    "handles": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "required": ["debug_handle", "fx_nodes"],
                            "properties": {
                                "debug_handle": {"type": "integer", "minimum": 0},
                                "fx_nodes": {
                                    "type": "array",
                                    "minItems": 1,
                                    "items": {"type": "string", "minLength": 1},
                                },
                            },
                        },
                    },
                    "ir_stages": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "required": ["name", "handles"],
                            "properties": {
                                "name": {"type": "string", "minLength": 1},
                                "handles": {
                                    "type": "array",
                                    "minItems": 1,
                                    "items": {
                                        "type": "integer",
                                        "minimum": 0,
                                    },
                                },
                            },
                        },
                    },
                    "profiler_events": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["name", "debug_handle"],
                            "properties": {
                                "name": {"type": "string", "minLength": 1},
                                "debug_handle": {"type": "integer", "minimum": 0},
                            },
                        },
                    },
                },
            },
        }
    },
}


def _validate_schema(instance, schema):
    if schema.get("type") == "object":
        assert isinstance(instance, dict)
        for key in schema.get("required", []):
            assert key in instance
        for key, subschema in schema.get("properties", {}).items():
            if key in instance:
                _validate_schema(instance[key], subschema)
        return
    if schema.get("type") == "array":
        assert isinstance(instance, list)
        if "minItems" in schema:
            assert len(instance) >= schema["minItems"]
        item_schema = schema.get("items")
        if item_schema is not None:
            for item in instance:
                _validate_schema(item, item_schema)
        return
    if schema.get("type") == "string":
        assert isinstance(instance, str)
        if "minLength" in schema:
            assert len(instance) >= schema["minLength"]
        return
    if schema.get("type") == "integer":
        assert type(instance) is int
        if "minimum" in schema:
            assert instance >= schema["minimum"]
        return
    raise AssertionError(f"Unsupported schema fragment: {schema}")


def _assert_kernel_provenance(kernel_record):
    handles = kernel_record["handles"]
    assert handles

    handle_ids = {entry["debug_handle"] for entry in handles}
    assert len(handle_ids) == len(handles)

    for handle in handles:
        assert handle["fx_nodes"]

    for stage in kernel_record["ir_stages"]:
        assert stage["handles"]
        assert set(stage["handles"]) == handle_ids

    for event in kernel_record.get("profiler_events", []):
        assert event["debug_handle"] in handle_ids

    kernel_name = kernel_record["compiled_kernel"]
    expected_handles = "-".join(str(handle_id) for handle_id in sorted(handle_ids))
    assert kernel_name.endswith(f"__h{expected_handles}")


def test_spyre_provenance_json_validates_against_schema(sample_provenance_payload):
    payload = json.loads(json.dumps(sample_provenance_payload))
    _validate_schema(payload, PROVENANCE_SCHEMA)


def test_spyre_provenance_tracks_each_ir_stage_without_gaps(sample_provenance_payload):
    assert sample_provenance_payload["kernels"]
    for kernel_record in sample_provenance_payload["kernels"]:
        _assert_kernel_provenance(kernel_record)


def test_spyre_provenance_negative_missing_handle_fails(sample_provenance_payload):
    broken_payload = json.loads(json.dumps(sample_provenance_payload))
    broken_payload["kernels"][0]["ir_stages"][1]["handles"] = [17]

    with pytest.raises(AssertionError):
        _assert_kernel_provenance(broken_payload["kernels"][0])


@pytest.mark.parametrize(
    ("compiled_kernel", "handle_ids"),
    [
        pytest.param("sdsc_linear__h7", [7], id="single_op"),
        pytest.param("sdsc_linear_relu__h17-23", [17, 23], id="fused_ops"),
        pytest.param(
            "sdsc_attention_block__h3-5-8",
            [3, 5, 8],
            id="attention_block",
        ),
    ],
)
def test_kernel_name_encoding_matches_expected_format(compiled_kernel, handle_ids):
    kernel_record = {
        "compiled_kernel": compiled_kernel,
        "handles": [
            {"debug_handle": handle_id, "fx_nodes": [f"node_{handle_id}"]}
            for handle_id in handle_ids
        ],
        "ir_stages": [{"name": "opspec", "handles": handle_ids}],
        "profiler_events": [],
    }

    _assert_kernel_provenance(kernel_record)
