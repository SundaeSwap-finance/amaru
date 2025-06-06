use amaru_kernel::{
    cbor, protocol_parameters::ProtocolParameters, Bytes, Hash, Hasher, MintedBlock, Point,
    PostAlonzoTransactionOutput, TransactionInput, TransactionOutput, Value, EraHistory,
    network::NetworkName
};
use amaru_ledger::{
    context,
    rules::{self},
    state::State,
};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};
use store::MemoryStore;

mod store;

type BlockWrapper<'b> = (u16, MintedBlock<'b>);

pub const RAW_BLOCK_CONWAY_1: &str = include_str!("../assets/conway1.block");
pub const RAW_BLOCK_CONWAY_3: &str = include_str!("../assets/conway3.block");

pub fn forward_ledger(raw_block: &str) {
    let bytes = hex::decode(raw_block).unwrap();

    let (_hash, block): BlockWrapper = cbor::decode(&bytes).unwrap();
    let era_history : &EraHistory = NetworkName::Preprod.into();

    let mut state = State::new(Arc::new(Mutex::new(MemoryStore {})), era_history);

    let point = Point::Specific(
        block.header.header_body.slot,
        Hasher::<256>::hash(block.header.raw_cbor()).to_vec(),
    );

    let issuer = Hasher::<224>::hash(&block.header.header_body.issuer_vkey[..]);

    fn create_input(transaction_id: &str, index: u64) -> TransactionInput {
        TransactionInput {
            transaction_id: Hash::from(hex::decode(transaction_id).unwrap().as_slice()),
            index,
        }
    }

    fn create_output(address: &str) -> TransactionOutput {
        TransactionOutput::PostAlonzo(PostAlonzoTransactionOutput {
            address: Bytes::from(hex::decode(address).unwrap()),
            value: Value::Coin(0),
            datum_option: None,
            script_ref: None,
        })
    }

    let utxos = BTreeMap::from([
        (
            create_input(
                "2e6b2226fd74ab0cadc53aaa18759752752bd9b616ea48c0e7b7be77d1af4bf4",
                0,
            ),
            create_output("61bbe56449ba4ee08c471d69978e01db384d31e29133af4546e6057335"),
        ),
        (
            create_input(
                "d5dc99581e5f479d006aca0cd836c2bb7ddcd4a243f8e9485d3c969df66462cb",
                0,
            ),
            create_output("61bbe56449ba4ee08c471d69978e01db384d31e29133af4546e6057335"),
        ),
    ]);

    let step = rules::validate_block(
        context::DefaultValidationContext::new(utxos),
        ProtocolParameters::default(),
        block,
    )
    .unwrap_or_else(|e| panic!("Failed to validate block: {:?}", e))
    .anchor(&point, issuer);

    state.forward(step).unwrap()
}
