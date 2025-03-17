use amaru_kernel::{
    protocol_parameters::ProtocolParameters, HasScriptHash, MintedTransactionBody,
    MintedWitnessSet, RequiresScript,
};

use crate::rules::{context::UtxoSlice, TransactionRuleViolation};

pub fn validate_script_integrity_hash(
    transaction_body: &MintedTransactionBody<'_>,
    witness_set: &MintedWitnessSet<'_>,
    protocol_parameters: &ProtocolParameters,
    utxo_slice: &UtxoSlice,
) -> Result<(), TransactionRuleViolation> {
    let required_spending_scripts = transaction_body
        .inputs
        .iter()
        .filter_map(|input| {
            utxo_slice
                .get(input)
                .and_then(|output| output.requires_script())
        })
        .collect::<Vec<_>>();

    let required_withdrawal_scripts = transaction_body
        .withdrawals
        .as_deref()
        .map(|withdrawals| {
            withdrawals
                .iter()
                .filter_map(|withdrawal| withdrawal.requires_script())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let required_certificate_scripts = transaction_body
        .certificates
        .as_deref()
        .map(|certificates| {
            certificates
                .iter()
                .filter_map(|certificate| certificate.requires_script())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let required_minting_scripts = transaction_body
        .mint
        .as_deref()
        .map(|multiasset| {
            multiasset
                .iter()
                .map(|(policy, _)| *policy)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let required_voting_scripts = transaction_body
        .voting_procedures
        .as_deref()
        .map(|voting_prodedures| {
            voting_prodedures
                .iter()
                .filter_map(|(voter, _)| voter.script_hash())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let required_proposing_scripts = transaction_body
        .proposal_procedures
        .as_deref()
        .map(|proposal_procedures| {
            proposal_procedures
                .iter()
                .filter_map(|proposal_procedure| proposal_procedure.requires_script())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let required_scripts = [
        required_spending_scripts,
        required_withdrawal_scripts,
        required_certificate_scripts,
        required_minting_scripts,
        required_voting_scripts,
        required_proposing_scripts,
    ]
    .concat();

    // scriptsProvided and scriptsNeeded
    // get langs from scriptsUsed (scriptsProvided filter scriptsNeeded.contains) -> scriptLanguage form script
    // langauge views set
    // build script integrity hash (lang views | redemeers | datums)
    // Check that our script integrity hash matches the provided script data hash
    todo!()
}
