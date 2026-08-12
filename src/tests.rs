use crate::*;

/// The `U256Balance` codec exists to keep the pre-codec layout byte-for-byte,
/// so that records written by earlier versions still decode. Pin the exact
/// offsets rather than just the round trip.
#[test]
fn u256_balance_reproduces_the_original_record_layout() {
    let addr = [7u8; 20];
    let balance = [9u8; 32];

    let record = record_of::<U256Balance>(&addr, &balance);
    assert_eq!(&record[..20], &addr);
    assert_eq!(&record[20..32], &[0u8; 12], "the historical padding");
    assert_eq!(&record[32..], &balance);
}

/// Every codec owns bytes 20..64 and nothing else; the address prefix is the
/// crate's, so a codec cannot break the not-in-set proof however it behaves.
#[test]
fn a_codec_cannot_reach_the_address_prefix() {
    /// Deliberately hostile: fills its whole payload with 0xff.
    struct Greedy;
    impl RecordCodec for Greedy {
        type Value = ();
        fn encode(_: &()) -> [u8; PAYLOAD_BYTES] {
            [0xffu8; PAYLOAD_BYTES]
        }
        fn decode(_: &[u8; PAYLOAD_BYTES]) {}
    }

    let addr = [3u8; 20];
    let record = record_of::<Greedy>(&addr, &());
    assert_eq!(&record[..20], &addr, "prefix survives a greedy codec");
    assert_eq!(&record[20..], &[0xffu8; PAYLOAD_BYTES]);
}

#[test]
fn a_codec_round_trips_through_a_record() {
    let addr = [1u8; 20];
    let balance = [0xabu8; 32];
    let record = record_of::<U256Balance>(&addr, &balance);
    assert_eq!(U256Balance::decode(payload_of(&record)), balance);
}

/// 20 address bytes plus the codec's payload must be exactly one record.
#[test]
fn payload_bytes_accounts_for_the_whole_record() {
    assert_eq!(20 + PAYLOAD_BYTES, std::mem::size_of::<Record>());
}

#[test]
fn default_shape_has_expected_capacity() {
    let (config, layout) = default_shape();
    assert_eq!(layout.num_payloads(config.column_height()), 33_554_432);
}

#[test]
fn client_bootstrap_rejects_invalid_directory_blob() {
    let err = match EthPirClient::<U256Balance>::try_new(b"not a keyword directory") {
        Ok(_) => panic!("invalid keyword directory was accepted"),
        Err(err) => err,
    };
    assert!(matches!(err, EthPirError::Io { .. }));
}

#[test]
fn io_error_mapping_preserves_truncated_wire_kind() {
    let err = match EthPirClient::<U256Balance>::new(b"not a keyword directory") {
        Ok(_) => panic!("invalid keyword directory was accepted"),
        Err(err) => err,
    };
    assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);
}
