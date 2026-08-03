use crate::*;

#[test]
fn record_layout_matches_contract() {
    let mut addr = [0u8; 20];
    addr.copy_from_slice(&[7u8; 20]);
    let mut balance = [0u8; 32];
    balance.copy_from_slice(&[9u8; 32]);

    let record = record_of(&addr, &balance);
    assert_eq!(&record[..20], &addr);
    assert_eq!(&record[20..32], &[0u8; 12]);
    assert_eq!(&record[32..], &balance);
    assert_eq!(address_slot(&addr), record[..32]);
}

#[test]
fn default_shape_has_expected_capacity() {
    let (config, layout) = default_shape();
    assert_eq!(layout.num_payloads(config.column_height()), 33_554_432);
}
