use crate::{Coin, Epoch, ExUnits, RationalNumber};

#[derive(Clone)]
pub struct ProtocolParameters {
    pub max_block_body_size: u32,
    pub max_tx_size: u32,
    pub max_header_size: u32,
    pub max_tx_ex_units: ExUnits,
    pub max_block_ex_units: ExUnits,
    pub max_val_size: u32,
    pub max_collateral_inputs: u32,

    pub min_fee_a: u32,
    pub min_fee_b: u32,
    pub key_deposit: Coin,
    pub pool_deposit: Coin,
    pub coins_per_utxo_byte: Coin,
    pub prices: Prices,
    pub min_fee_ref_script_coins_per_byte: RationalNumber,
    pub max_ref_script_size_per_tx: u32,
    pub max_ref_script_size_per_block: u32,
    pub ref_script_cost_stride: u32,
    pub ref_script_cost_multiplier: RationalNumber,

    pub max_epoch: Epoch,
    pub pool_influence: RationalNumber,
    pub desired_pool_count: u32,
    pub treasury_expansion_rate: RationalNumber,
    pub monetary_expansion_rate: RationalNumber,
    pub collateral_percentage: u32,
    pub cost_models: CostModels,

    pub pool_thresholds: PoolThresholds,
    pub drep_thresholds: DrepThresholds,
    pub cc_min_size: u32,
    pub cc_max_term_length: u32,
    pub gov_action_lifetime: u32,
    pub gov_action_deposit: Coin,
    pub drep_deposit: Coin,
    pub drep_activity: Epoch,
}

#[derive(Debug, Clone)]
pub struct Prices {
    pub mem: RationalNumber,
    pub step: RationalNumber,
}

#[derive(Debug, Clone)]
pub struct CostModels {
    pub plutus_v1: Vec<i64>,
    pub plutus_v2: Vec<i64>,
    pub plutus_v3: Vec<i64>,
}

#[derive(Debug, Clone)]
pub struct PoolThresholds {
    // named `q1` in the spec
    pub no_confidence: RationalNumber,
    // named `q2a` in the spec
    pub committee: RationalNumber,
    // named `q2b` in the spec
    pub committee_under_no_confidence: RationalNumber,
    // named `q4` in the spec
    pub hard_fork: RationalNumber,
    // named `q5e` in the spec
    pub security_group: RationalNumber,
}

#[derive(Debug, Clone)]
pub struct DrepThresholds {
    // named `p1` in the spec
    pub no_confidence: RationalNumber,
    // named `p2` in the spec
    pub committee: RationalNumber,
    // named `p2b` in the spec
    pub committee_under_no_confidence: RationalNumber,
    // named `p3` in the spec
    pub constitution: RationalNumber,
    // named `p4` in the spec
    pub hard_fork: RationalNumber,
    pub protocol_parameters: ProtocolParametersThresholds,
    // named `p6` in the spec
    pub treasury_withdrawal: RationalNumber,
}

#[derive(Debug, Clone)]
pub struct ProtocolParametersThresholds {
    // named `p5a` in the spec
    pub network_group: RationalNumber,
    // named `p5b` in the spec
    pub economic_group: RationalNumber,
    // named `p5c` in the spec
    pub technical_group: RationalNumber,
    // named `p5d` in the spec
    pub governance_group: RationalNumber,
}

impl Default for ProtocolParameters {
    // This default is the protocol parameters on Preprod as of epoch 197
    fn default() -> Self {
        Self {
            min_fee_a: 44,
            min_fee_b: 155381,
            max_block_body_size: 90112,
            max_tx_size: 16384,
            max_header_size: 1100,
            max_tx_ex_units: ExUnits {
                mem: 14000000,
                steps: 10000000000,
            },
            max_block_ex_units: ExUnits {
                mem: 62000000,
                steps: 20000000000,
            },
            max_val_size: 5000,
            max_collateral_inputs: 3,
            key_deposit: 2000000,
            pool_deposit: 500000000,
            coins_per_utxo_byte: 4310,
            prices: Prices {
                mem: RationalNumber {
                    numerator: 577,
                    denominator: 10000,
                },
                step: RationalNumber {
                    numerator: 721,
                    denominator: 10000000,
                },
            },
            min_fee_ref_script_coins_per_byte: RationalNumber {
                numerator: 15,
                denominator: 1,
            },
            max_ref_script_size_per_tx: 200 * 1024, //Hardcoded in the haskell ledger (https://github.com/IntersectMBO/cardano-ledger/blob/3fe73a26588876bbf033bf4c4d25c97c2d8564dd/eras/conway/impl/src/Cardano/Ledger/Conway/Rules/Ledger.hs#L154)
            max_ref_script_size_per_block: 1024 * 1024, // Hardcoded in the haskell ledger (https://github.com/IntersectMBO/cardano-ledger/blob/3fe73a26588876bbf033bf4c4d25c97c2d8564dd/eras/conway/impl/src/Cardano/Ledger/Conway/Rules/Bbody.hs#L91)
            ref_script_cost_stride: 25600, // Hardcoded in the haskell ledger (https://github.com/IntersectMBO/cardano-ledger/blob/3fe73a26588876bbf033bf4c4d25c97c2d8564dd/eras/conway/impl/src/Cardano/Ledger/Conway/Tx.hs#L82)
            ref_script_cost_multiplier: RationalNumber {
                numerator: 12,
                denominator: 10,
            }, // Hardcoded in the haskell ledger (https://github.com/IntersectMBO/cardano-ledger/blob/3fe73a26588876bbf033bf4c4d25c97c2d8564dd/eras/conway/impl/src/Cardano/Ledger/Conway/Tx.hs#L85)
            max_epoch: 18,
            pool_influence: RationalNumber {
                numerator: 3,
                denominator: 10,
            },
            desired_pool_count: 500,
            treasury_expansion_rate: RationalNumber {
                numerator: 2,
                denominator: 10,
            },
            monetary_expansion_rate: RationalNumber {
                numerator: 3,
                denominator: 1000,
            },
            collateral_percentage: 150,
            cost_models: CostModels {
                plutus_v1: vec![
                    100788, 420, 1, 1, 1000, 173, 0, 1, 1000, 59957, 4, 1, 11183, 32, 201305, 8356,
                    4, 16000, 100, 16000, 100, 16000, 100, 16000, 100, 16000, 100, 16000, 100, 100,
                    100, 16000, 100, 94375, 32, 132994, 32, 61462, 4, 72010, 178, 0, 1, 22151, 32,
                    91189, 769, 4, 2, 85848, 228465, 122, 0, 1, 1, 1000, 42921, 4, 2, 24548, 29498,
                    38, 1, 898148, 27279, 1, 51775, 558, 1, 39184, 1000, 60594, 1, 141895, 32,
                    83150, 32, 15299, 32, 76049, 1, 13169, 4, 22100, 10, 28999, 74, 1, 28999, 74,
                    1, 43285, 552, 1, 44749, 541, 1, 33852, 32, 68246, 32, 72362, 32, 7243, 32,
                    7391, 32, 11546, 32, 85848, 228465, 122, 0, 1, 1, 90434, 519, 0, 1, 74433, 32,
                    85848, 228465, 122, 0, 1, 1, 85848, 228465, 122, 0, 1, 1, 270652, 22588, 4,
                    1457325, 64566, 4, 20467, 1, 4, 0, 141992, 32, 100788, 420, 1, 1, 81663, 32,
                    59498, 32, 20142, 32, 24588, 32, 20744, 32, 25933, 32, 24623, 32, 53384111,
                    14333, 10,
                ],
                plutus_v2: vec![
                    100788, 420, 1, 1, 1000, 173, 0, 1, 1000, 59957, 4, 1, 11183, 32, 201305, 8356,
                    4, 16000, 100, 16000, 100, 16000, 100, 16000, 100, 16000, 100, 16000, 100, 100,
                    100, 16000, 100, 94375, 32, 132994, 32, 61462, 4, 72010, 178, 0, 1, 22151, 32,
                    91189, 769, 4, 2, 85848, 228465, 122, 0, 1, 1, 1000, 42921, 4, 2, 24548, 29498,
                    38, 1, 898148, 27279, 1, 51775, 558, 1, 39184, 1000, 60594, 1, 141895, 32,
                    83150, 32, 15299, 32, 76049, 1, 13169, 4, 22100, 10, 28999, 74, 1, 28999, 74,
                    1, 43285, 552, 1, 44749, 541, 1, 33852, 32, 68246, 32, 72362, 32, 7243, 32,
                    7391, 32, 11546, 32, 85848, 228465, 122, 0, 1, 1, 90434, 519, 0, 1, 74433, 32,
                    85848, 228465, 122, 0, 1, 1, 85848, 228465, 122, 0, 1, 1, 955506, 213312, 0, 2,
                    270652, 22588, 4, 1457325, 64566, 4, 20467, 1, 4, 0, 141992, 32, 100788, 420,
                    1, 1, 81663, 32, 59498, 32, 20142, 32, 24588, 32, 20744, 32, 25933, 32, 24623,
                    32, 43053543, 10, 53384111, 14333, 10, 43574283, 26308, 10,
                ],
                plutus_v3: vec![
                    100788, 420, 1, 1, 1000, 173, 0, 1, 1000, 59957, 4, 1, 11183, 32, 201305, 8356,
                    4, 16000, 100, 16000, 100, 16000, 100, 16000, 100, 16000, 100, 16000, 100, 100,
                    100, 16000, 100, 94375, 32, 132994, 32, 61462, 4, 72010, 178, 0, 1, 22151, 32,
                    91189, 769, 4, 2, 85848, 123203, 7305, -900, 1716, 549, 57, 85848, 0, 1, 1,
                    1000, 42921, 4, 2, 24548, 29498, 38, 1, 898148, 27279, 1, 51775, 558, 1, 39184,
                    1000, 60594, 1, 141895, 32, 83150, 32, 15299, 32, 76049, 1, 13169, 4, 22100,
                    10, 28999, 74, 1, 28999, 74, 1, 43285, 552, 1, 44749, 541, 1, 33852, 32, 68246,
                    32, 72362, 32, 7243, 32, 7391, 32, 11546, 32, 85848, 123203, 7305, -900, 1716,
                    549, 57, 85848, 0, 1, 90434, 519, 0, 1, 74433, 32, 85848, 123203, 7305, -900,
                    1716, 549, 57, 85848, 0, 1, 1, 85848, 123203, 7305, -900, 1716, 549, 57, 85848,
                    0, 1, 955506, 213312, 0, 2, 270652, 22588, 4, 1457325, 64566, 4, 20467, 1, 4,
                    0, 141992, 32, 100788, 420, 1, 1, 81663, 32, 59498, 32, 20142, 32, 24588, 32,
                    20744, 32, 25933, 32, 24623, 32, 43053543, 10, 53384111, 14333, 10, 43574283,
                    26308, 10, 16000, 100, 16000, 100, 962335, 18, 2780678, 6, 442008, 1, 52538055,
                    3756, 18, 267929, 18, 76433006, 8868, 18, 52948122, 18, 1995836, 36, 3227919,
                    12, 901022, 1, 166917843, 4307, 36, 284546, 36, 158221314, 26549, 36, 74698472,
                    36, 333849714, 1, 254006273, 72, 2174038, 72, 2261318, 64571, 4, 207616, 8310,
                    4, 1293828, 28716, 63, 0, 1, 1006041, 43623, 251, 0, 1, 100181, 726, 719, 0, 1,
                    100181, 726, 719, 0, 1, 100181, 726, 719, 0, 1, 107878, 680, 0, 1, 95336, 1,
                    281145, 18848, 0, 1, 180194, 159, 1, 1, 158519, 8942, 0, 1, 159378, 8813, 0, 1,
                    107490, 3298, 1, 106057, 655, 1, 1964219, 24520, 3,
                ],
            },
            pool_thresholds: PoolThresholds {
                no_confidence: RationalNumber {
                    numerator: 51,
                    denominator: 100,
                },
                committee: RationalNumber {
                    numerator: 51,
                    denominator: 100,
                },
                committee_under_no_confidence: RationalNumber {
                    numerator: 51,
                    denominator: 100,
                },
                hard_fork: RationalNumber {
                    numerator: 51,
                    denominator: 100,
                },
                security_group: RationalNumber {
                    numerator: 51,
                    denominator: 100,
                },
            },
            drep_thresholds: DrepThresholds {
                no_confidence: RationalNumber {
                    numerator: 51,
                    denominator: 100,
                },
                committee: RationalNumber {
                    numerator: 67,
                    denominator: 100,
                },
                committee_under_no_confidence: RationalNumber {
                    numerator: 67,
                    denominator: 100,
                },
                constitution: RationalNumber {
                    numerator: 6,
                    denominator: 10,
                },
                hard_fork: RationalNumber {
                    numerator: 75,
                    denominator: 100,
                },
                protocol_parameters: ProtocolParametersThresholds {
                    network_group: RationalNumber {
                        numerator: 6,
                        denominator: 10,
                    },
                    economic_group: RationalNumber {
                        numerator: 67,
                        denominator: 100,
                    },
                    technical_group: RationalNumber {
                        numerator: 67,
                        denominator: 100,
                    },
                    governance_group: RationalNumber {
                        numerator: 75,
                        denominator: 100,
                    },
                },
                treasury_withdrawal: RationalNumber {
                    numerator: 67,
                    denominator: 100,
                },
            },
            cc_min_size: 7,
            cc_max_term_length: 146,
            gov_action_lifetime: 6,
            gov_action_deposit: 100000000000,
            drep_deposit: 500000000,
            drep_activity: 20,
        }
    }
}
