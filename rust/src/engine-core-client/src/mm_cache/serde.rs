// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Writer-side serialization for the shm object storage.
//!
//! Rust port of the `MultiModalKwargsItem` branch of `MsgpackSerde.serialize`
//! from `vllm/distributed/device_communicators/shm_object_storage.py`. Only
//! the serialize direction is needed: deserialization lives in the Python
//! receiver cache.

use bytes::Bytes;

use crate::error::Result;
use crate::protocol::encode_msgpack;
use crate::protocol::multimodal::{MmKwargsItem, extract_mm_kwargs_item_aux_frames};

/// Matches `DEFAULT_MSGPACK_ZERO_COPY_THRESHOLD` used by the ZMQ client path:
/// tensor buffers at or above this size are moved into auxiliary frames.
const AUX_FRAME_THRESHOLD: usize = 256;

/// Pickled metadata type name for a serialized `MultiModalKwargsItem`.
const ITEM_TYPE_NAME: &[u8] = b"MultiModalKwargsItem";

/// A serialized `MmKwargsItem`: the msgpack body frame followed by the
/// auxiliary tensor frames, plus the pickled object metadata.
pub(crate) struct SerializedMmItem {
    /// `[msgpack body, aux frame 1, aux frame 2, ...]`.
    pub frames: Vec<Bytes>,
    /// Pickled `("MultiModalKwargsItem", nbytes, len_arr)` metadata.
    pub metadata: Vec<u8>,
}

/// Serialize an item exactly like the Rust ZMQ path: clone the item, extract
/// large tensor buffers into auxiliary frames, then encode the body with
/// `rmp_serde::to_vec_named`. The Python decoder splits the payload by
/// `len_arr` and feeds the frames to `MsgpackDecoder(MultiModalKwargsItem)`.
pub(crate) fn serialize_mm_item(item: &MmKwargsItem) -> Result<SerializedMmItem> {
    let mut item = item.clone();
    let mut aux_frames = Vec::new();
    extract_mm_kwargs_item_aux_frames(&mut item, &mut aux_frames, AUX_FRAME_THRESHOLD);

    let body = Bytes::from(encode_msgpack(&item)?);
    let mut frames = vec![body];
    frames.extend(aux_frames);

    let len_arr = frames.iter().map(|frame| frame.len() as u64).collect::<Vec<_>>();
    let nbytes = len_arr.iter().sum();
    let metadata = encode_item_metadata(nbytes, &len_arr);
    Ok(SerializedMmItem { frames, metadata })
}

/// Hand-written protocol-2 pickle byte stream of the object metadata tuple
/// `(type_name, nbytes, len_arr)`.
///
/// Python writes `pickle.dumps(metadata, protocol=pickle.HIGHEST_PROTOCOL)`,
/// but `pickle.loads` auto-detects the protocol version, so protocol-2 bytes
/// are equivalent on the read side.
///
/// Layout: `\x80\x02` (PROTO 2), `(` MARK, SHORT_BINUNICODE type name
/// (`\x8c` + len byte + bytes), the int nbytes, `]` EMPTY_LIST, `(` MARK,
/// each int, `e` APPENDS, `t` TUPLE, `.` STOP.
fn encode_item_metadata(nbytes: u64, len_arr: &[u64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(16 + ITEM_TYPE_NAME.len() + 5 * (1 + len_arr.len()));
    out.extend_from_slice(&[0x80, 0x02, b'(']);
    out.push(0x8c);
    out.push(ITEM_TYPE_NAME.len() as u8);
    out.extend_from_slice(ITEM_TYPE_NAME);
    encode_pickle_int(&mut out, nbytes);
    out.extend_from_slice(b"](");
    for &len in len_arr {
        encode_pickle_int(&mut out, len);
    }
    out.extend_from_slice(b"et.");
    out
}

/// Pickle BININT1 (`K` + 1 byte) for values < 256, else BININT (`J` + 4-byte
/// LE i32). Larger values are impossible here (max object size is 128 MiB).
fn encode_pickle_int(out: &mut Vec<u8>, value: u64) {
    if value < 256 {
        out.extend_from_slice(&[b'K', value as u8]);
    } else {
        assert!(
            value <= i32::MAX as u64,
            "metadata int out of range: {value}"
        );
        out.push(b'J');
        out.extend_from_slice(&(value as i32).to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::protocol::multimodal::{MmBatchedField, MmField, MmFieldElem, MmKwargValue};
    use crate::protocol::tensor::WireTensor;

    fn int_elem(value: i64) -> MmFieldElem {
        MmFieldElem {
            data: Some(MmKwargValue::Int(value)),
            field: MmField::Batched(MmBatchedField { keep_on_cpu: false }),
        }
    }

    #[test]
    fn pickle_metadata_matches_expected_byte_stream() {
        // `("MultiModalKwargsItem", 560, [48, 512])` in protocol 2.
        let metadata = encode_item_metadata(560, &[48, 512]);
        expect_test::expect![[r#"8002288c144d756c74694d6f64616c4b77617267734974656d4a300200005d284b304a0002000065742e"#]]
            .assert_eq(&hex::encode(&metadata));
    }

    #[test]
    fn pickle_metadata_uses_binint1_below_256() {
        let metadata = encode_item_metadata(3, &[3]);
        expect_test::expect![[
            r#"8002288c144d756c74694d6f64616c4b77617267734974656d4b035d284b0365742e"#
        ]]
        .assert_eq(&hex::encode(metadata));
    }

    #[test]
    fn serialize_splits_large_tensors_into_aux_frames() {
        let big_data = (0..128).map(|value| value as f32).collect::<Vec<_>>();
        let big_bytes = big_data.iter().flat_map(|value| value.to_ne_bytes()).collect::<Vec<_>>();
        let item: MmKwargsItem = BTreeMap::from([
            ("offset".to_string(), int_elem(7)),
            (
                "pixel_values".to_string(),
                MmFieldElem {
                    data: Some(MmKwargValue::Tensor(
                        WireTensor::from_f32(vec![128], big_data).unwrap(),
                    )),
                    field: MmField::Batched(MmBatchedField { keep_on_cpu: false }),
                },
            ),
            (
                "small".to_string(),
                MmFieldElem {
                    data: Some(MmKwargValue::Tensor(
                        WireTensor::from_f32(vec![2], vec![1.0, 2.0]).unwrap(),
                    )),
                    field: MmField::Batched(MmBatchedField { keep_on_cpu: false }),
                },
            ),
        ]);

        let serialized = serialize_mm_item(&item).unwrap();
        assert_eq!(serialized.frames.len(), 2);
        assert_eq!(serialized.frames[1].as_ref(), big_bytes.as_slice());

        // The body references the big tensor as aux frame index 1 while the
        // small tensor stays inline as a raw-view ext.
        let body = crate::protocol::decode_value(&serialized.frames[0]).unwrap();
        let map = body.as_map().expect("item body is a map");
        let field_value = |key: &str| {
            map.iter()
                .find(|(k, _)| k.as_str() == Some(key))
                .unwrap_or_else(|| panic!("missing key {key}"))
                .1
                .clone()
        };
        let pixel_values = field_value("pixel_values");
        let big_tuple = pixel_values.as_map().unwrap();
        let data = big_tuple
            .iter()
            .find(|(k, _)| k.as_str() == Some("data"))
            .map(|(_, v)| v)
            .expect("data field");
        let tensor_tuple = data.as_array().expect("tensor tuple");
        assert_eq!(tensor_tuple[2].as_u64(), Some(1));

        let small = field_value("small");
        let small_data = small
            .as_map()
            .unwrap()
            .iter()
            .find(|(k, _)| k.as_str() == Some("data"))
            .map(|(_, v)| v.clone())
            .unwrap();
        let small_tuple = small_data.as_array().expect("small tensor tuple");
        assert_eq!(
            small_tuple[2],
            rmpv::Value::Ext(
                3,
                [1.0_f32, 2.0].into_iter().flat_map(f32::to_ne_bytes).collect(),
            )
        );

        // len_arr = [body len, aux len]; nbytes = sum(len_arr).
        let total = serialized.frames.iter().map(|f| f.len() as u64).sum::<u64>();
        let metadata = encode_item_metadata(total, &[serialized.frames[0].len() as u64, 512]);
        assert_eq!(serialized.metadata, metadata);
    }
}
