// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Configurable Asset Privacy (CAP) library.

// This program is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version. This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details. You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Generation and verification of user configurable transfer notes
use crate::{
    errors::TxnApiError,
    keys::UserKeyPair,
    prelude::CapConfig,
    proof::transfer::{
        TransferProvingKey, TransferPublicInput, TransferVerifyingKey, TransferWitness,
    },
    structs::{
        Amount, AssetCode, AssetDefinition, ExpirableCredential, FeeInput, FreezeFlag, Nullifier,
        RecordCommitment, RecordOpening, TxnFeeInfo, ViewableMemo,
    },
    utils::{
        safe_sum_amount,
        txn_helpers::{transfer::*, *},
    },
};
use ark_serialize::*;
use ark_std::{
    format,
    rand::{CryptoRng, RngCore},
    string::ToString,
    vec,
    vec::Vec,
};
use jf_plonk::proof_system::structs::Proof;
use jf_primitives::{
    merkle_tree::{AccMemberWitness, NodeValue},
    signatures::schnorr,
};
use serde::{Deserialize, Serialize};

/// Anonymous Transfer note structure for single sender, single asset type (+
/// native asset type for fees)
#[derive(CanonicalSerialize, CanonicalDeserialize, Serialize, Deserialize, Derivative)]
#[derivative(
    Debug(bound = "C: CapConfig"),
    Clone(bound = "C: CapConfig"),
    PartialEq(bound = "C: CapConfig"),
    Eq(bound = "C: CapConfig"),
    Hash(bound = "C: CapConfig")
)]
pub struct TransferNote<C: CapConfig> {
    /// nullifier for inputs
    pub inputs_nullifiers: Vec<Nullifier<C>>,
    /// generated output commitments
    pub output_commitments: Vec<RecordCommitment<C>>,
    /// proof of spending and policy compliance
    pub proof: Proof<C::PairingCurve>,
    /// Memo generated for policy compliance
    pub viewing_memo: ViewableMemo<C>,
    /// Auxiliary information (merkle root, native asset, fee, valid max time,
    /// receiver memos verification key)
    pub aux_info: AuxInfo<C>,
}

/// Auxiliary info of TransferNote: includes merkle root, native asset, fee,
/// valid max time and receiver memos verification key
#[derive(CanonicalSerialize, CanonicalDeserialize, Serialize, Deserialize, Derivative)]
#[derivative(
    Debug(bound = "C: CapConfig"),
    Clone(bound = "C: CapConfig"),
    PartialEq(bound = "C: CapConfig"),
    Eq(bound = "C: CapConfig"),
    Hash(bound = "C: CapConfig")
)]
pub struct AuxInfo<C: CapConfig> {
    /// Accumulator state
    pub merkle_root: NodeValue<C::ScalarField>,
    /// proposed fee in native asset type for the transfer
    pub fee: Amount,
    /// A projected future timestamp that the snark proof should be valid until,
    /// especially for credential to still hold valid/unexpired
    pub valid_until: u64,
    /// Transaction memos signature verification key (usually used for signing
    /// receiver memos)
    pub txn_memo_ver_key: schnorr::VerKey<C::EmbeddedCurveParam>,
    /// Additional data bound to `TransferValidityProof`
    pub extra_proof_bound_data: Vec<u8>,
}

/// All necessary information for each input record in the `TransferNote`
/// generation.
#[derive(Derivative)]
#[derivative(Debug(bound = "C: CapConfig"))]
pub struct TransferNoteInput<'kp, C: CapConfig> {
    /// Record opening of the input record.
    pub ro: RecordOpening<C>,
    /// Accumulator membership proof (namely the Merkle proof) of the record
    /// commitment.
    pub acc_member_witness: AccMemberWitness<C::ScalarField>,
    /// Reference of the record owner's key pair.
    pub owner_keypair: &'kp UserKeyPair<C>,
    /// The identity credential of the user. Optional, only needed if asset
    /// policy has a non-empty `ViewerPubKey`.
    pub cred: Option<ExpirableCredential<C>>,
}

impl<'kp, C: CapConfig> Clone for TransferNoteInput<'kp, C> {
    fn clone(&self) -> Self {
        Self {
            ro: self.ro.clone(),
            acc_member_witness: self.acc_member_witness.clone(),
            owner_keypair: self.owner_keypair,
            cred: self.cred.clone(),
        }
    }
}

impl<'kp, C: CapConfig> From<FeeInput<'kp, C>> for TransferNoteInput<'kp, C> {
    fn from(input: FeeInput<'kp, C>) -> Self {
        Self {
            ro: input.ro,
            acc_member_witness: input.acc_member_witness,
            owner_keypair: input.owner_keypair,
            cred: None,
        }
    }
}

impl<C: CapConfig> TransferNote<C> {
    /// Generate a note for transfering native asset
    ///
    /// * `inputs` - list of input records and associated witness to spend them,
    /// sum of their values should be >= sum of values of `outputs`
    /// * `outputs` - list of record opening for receivers (**not including the
    ///   fee change record**)
    /// * `valid_until` - A projected future timestamp that the proof should be
    ///   valid until, specifically for credential expiry
    /// * `proving_key` - proving key used to generate validity proof
    ///
    /// Returns: (transfer note, signature key to bind a message to the transfer
    /// note proof, Record opening of fee change directed at first input
    /// address) tuple on successful generation.
    #[allow(clippy::type_complexity)]
    pub fn generate_native<R: CryptoRng + RngCore>(
        rng: &mut R,
        inputs: Vec<TransferNoteInput<C>>,
        outputs: &[RecordOpening<C>],
        fee: Amount,
        valid_until: u64,
        proving_key: &TransferProvingKey<C>,
    ) -> Result<
        (
            Self,
            schnorr::KeyPair<C::EmbeddedCurveParam>,
            RecordOpening<C>,
        ),
        TxnApiError,
    >
    where
        R: CryptoRng + RngCore,
    {
        if !inputs[0].ro.asset_def.is_native()
            || inputs
                .iter()
                .skip(1)
                .any(|input| !(input.ro.is_dummy() || input.ro.asset_def.is_native()))
            || outputs.iter().any(|output| !output.asset_def.is_native())
        {
            return Err(TxnApiError::InvalidParameter(
                "Should only contain native asset types in inputs and outputs,\
                 if you are trying to transfer non-native assets, \
                 please use `TransferNote::generate_non_native()` api."
                    .to_string(),
            ));
        }
        let in_amounts: Vec<_> = inputs
            .iter()
            .filter(|input| !input.ro.is_dummy())
            .map(|input| input.ro.amount)
            .collect();
        let total_in = safe_sum_amount(&in_amounts).ok_or_else(|| {
            TxnApiError::InvalidParameter(
                "Total input amount exceeds max value (2^64-1)".to_string(),
            )
        })?;
        let out_amounts: Vec<_> = outputs.iter().map(|output| output.amount).collect();
        let total_out = safe_sum_amount(&out_amounts).ok_or_else(|| {
            TxnApiError::InvalidParameter(
                "Total output amount exceeds max value (2^64-1)".to_string(),
            )
        })?;

        if total_in < total_out + fee {
            return Err(TxnApiError::InvalidParameter(
                "Total amount inputs should be at least total amont output + fee".to_string(),
            ));
        }
        let fee_change = total_in - total_out - fee;

        // get fee change public key, we check later that input[0] is not dummy
        let fee_chg_pub_key = inputs[0].ro.pub_key.clone();

        let fee_change_ro = RecordOpening::new(
            rng,
            fee_change,
            AssetDefinition::native(),
            fee_chg_pub_key,
            FreezeFlag::Unfrozen,
        );
        let mut outputs_with_fee_change = vec![fee_change_ro.clone()];
        outputs_with_fee_change.extend_from_slice(outputs);
        let (note, sig_key) = Self::generate(
            rng,
            inputs,
            &outputs_with_fee_change,
            proving_key,
            valid_until,
            vec![],
        )?;
        Ok((note, sig_key, fee_change_ro))
    }

    /// Generate a note for transfering non-native asset
    ///
    /// * `rng` - Randomness generator
    /// * `inputs` - list of input records and associated witness to spend them,
    /// sum of their values should be >= sum of values of `outputs`
    /// * `outputs` - list of record opening for receivers (**excluding the fee
    ///   change record**)
    /// * `fee_input` - the input record and associated witness for fee payment
    /// * `fee` - concrete amount of fee to pay, <= value of `fee_input`
    /// * `valid_until` - A projected future timestamp that the proof should be
    ///   valid until, specifically for credential expiry
    /// * `proving_key` - proving key used to generate validity proof
    /// * `extra_proof_bound_data` - additional data bound to validity proof
    ///   (thus bound to the transaction)
    ///
    /// Returns: (transfer note, signature key to bind a message to the transfer
    /// note proof, fee change record opening) tuple on successful
    /// generation.
    pub fn generate_non_native<R: CryptoRng + RngCore>(
        rng: &mut R,
        inputs: Vec<TransferNoteInput<C>>,
        outputs: &[RecordOpening<C>],
        fee: TxnFeeInfo<C>,
        valid_until: u64,
        proving_key: &TransferProvingKey<C>,
        extra_proof_bound_data: Vec<u8>,
    ) -> Result<(Self, schnorr::KeyPair<C::EmbeddedCurveParam>), TxnApiError>
    where
        R: CryptoRng + RngCore,
    {
        check_fee(&fee)?;
        let mut fee_prepended_inputs = vec![fee.fee_input.into()];
        fee_prepended_inputs.extend_from_slice(&inputs[..]);
        let outputs = [&[fee.fee_chg_ro], outputs].concat();

        Self::generate(
            rng,
            fee_prepended_inputs,
            &outputs,
            proving_key,
            valid_until,
            extra_proof_bound_data,
        )
    }

    /// Generates a transfer note
    /// * `rng` - Randomness generator
    /// * `inputs` - Input record openings and all necessary witness
    /// * `outputs` - Input record openings
    /// * `proving_key` - Prover parameters
    /// * `valid_until` - A projected future timestamp that the proof should be
    ///   valid until, specifically for credential expiry
    /// On success returns triple:
    ///  Generated transfer note
    ///  Receivers' memos
    ///  Signature over produced receivers' memos
    /// On error return TxnApIError
    fn generate<R: CryptoRng + RngCore>(
        rng: &mut R,
        inputs: Vec<TransferNoteInput<C>>,
        outputs: &[RecordOpening<C>],
        proving_key: &TransferProvingKey<C>,
        valid_until: u64,
        extra_proof_bound_data: Vec<u8>,
    ) -> Result<(Self, schnorr::KeyPair<C::EmbeddedCurveParam>), TxnApiError> {
        // 1. check input correctness
        if inputs.is_empty() || outputs.is_empty() {
            return Err(TxnApiError::InvalidParameter(
                "input records and output records should NOT be empty".to_string(),
            ));
        }
        let input_ros: Vec<&RecordOpening<C>> = inputs.iter().map(|input| &input.ro).collect();
        let output_refs: Vec<&RecordOpening<C>> = outputs.iter().collect();
        check_proving_key_consistency(proving_key, &inputs, outputs.len())?;
        check_input_pub_keys(&inputs)?;
        check_dummy_inputs(&input_ros)?;
        let fee = check_balance(&input_ros, &output_refs)?;
        check_asset_def(&input_ros, &output_refs)?;
        check_unfrozen(&input_ros, &output_refs)?;
        let merkle_root = check_and_get_roots(&inputs)?;
        check_creds(&inputs, valid_until)?;

        // 2. build public input and snark proof
        let signing_keypair = schnorr::KeyPair::generate(rng);
        let witness = TransferWitness::new_unchecked(rng, inputs, outputs)?;
        let pub_inputs = TransferPublicInput::from_witness(&witness, valid_until)?;
        check_distinct_input_nullifiers(&pub_inputs.input_nullifiers)?;

        let proof = crate::proof::transfer::prove(
            rng,
            proving_key,
            &witness,
            &pub_inputs,
            signing_keypair.ver_key_ref(),
            &extra_proof_bound_data,
        )?;

        let transfer_note = TransferNote {
            inputs_nullifiers: pub_inputs.input_nullifiers,
            output_commitments: pub_inputs.output_commitments,
            proof,
            viewing_memo: pub_inputs.viewing_memo,
            aux_info: AuxInfo {
                merkle_root,
                fee,
                valid_until,
                txn_memo_ver_key: signing_keypair.ver_key(),
                extra_proof_bound_data,
            },
        };

        Ok((transfer_note, signing_keypair))
    }

    /// Generates a transfer note without native asset type
    /// * `rng` - Randomness generator
    /// * `inputs` - Input record openings and all necessary witness
    /// * `outputs` - Input record openings
    /// * `proving_key` - Prover parameters
    /// * `valid_until` - A projected future timestamp that the proof should be
    ///   valid until, specifically for credential expiry
    /// On success returns triple:
    ///  Generated transfer note
    ///  Receivers' memos
    ///  Signature over produced receivers' memos
    /// On error return TxnApIError
    /// * `rng` - Randomness generator
    /// * `inputs` - Input record openings and all necessary witness
    /// * `outputs` - Input record openings
    /// * `proving_key` - Prover parameters
    /// * `valid_until` - A projected future timestamp that the proof should be
    ///   valid until, specifically for credential expiry
    /// On success returns triple:
    ///  Generated transfer note
    ///  Receivers' memos
    ///  Signature over produced receivers' memos
    /// On error return TxnApIError
    #[cfg(feature = "transfer_non_native_fee")]
    pub fn generate_without_native<R: CryptoRng + RngCore>(
        rng: &mut R,
        inputs: Vec<TransferNoteInput<C>>,
        outputs: &[RecordOpening<C>],
        proving_key: &TransferProvingKey<C>,
        valid_until: u64,
        extra_proof_bound_data: Vec<u8>,
    ) -> Result<(Self, schnorr::KeyPair<C::EmbeddedCurveParam>), TxnApiError> {
        // 1. check input correctness
        if inputs.is_empty() || outputs.is_empty() {
            return Err(TxnApiError::InvalidParameter(
                "input records and output records should NOT be empty".to_string(),
            ));
        }
        let input_ros: Vec<&RecordOpening<C>> = inputs.iter().map(|input| &input.ro).collect();
        let output_refs: Vec<&RecordOpening<C>> = outputs.iter().collect();
        check_proving_key_consistency(proving_key, &inputs, outputs.len())?;
        check_input_pub_keys(&inputs)?;
        check_dummy_inputs(&input_ros)?;
        check_non_native_asset_def(&input_ros, &output_refs)?;
        let fee = check_non_native_balance(&input_ros, &output_refs)?;
        check_unfrozen(&input_ros, &output_refs)?;
        let merkle_root = check_and_get_roots(&inputs)?;
        check_creds(&inputs, valid_until)?;

        // 2. build public input and snark proof
        let signing_keypair = schnorr::KeyPair::generate(rng);
        let witness = TransferWitness::new_unchecked(rng, inputs, outputs)?;
        let pub_inputs = TransferPublicInput::from_witness(&witness, valid_until)?;
        check_distinct_input_nullifiers(&pub_inputs.input_nullifiers)?;

        let proof = crate::proof::transfer::prove(
            rng,
            proving_key,
            &witness,
            &pub_inputs,
            signing_keypair.ver_key_ref(),
            &extra_proof_bound_data,
        )?;

        let transfer_note = TransferNote {
            inputs_nullifiers: pub_inputs.input_nullifiers,
            output_commitments: pub_inputs.output_commitments,
            proof,
            viewing_memo: pub_inputs.viewing_memo,
            aux_info: AuxInfo {
                merkle_root,
                fee,
                valid_until,
                txn_memo_ver_key: signing_keypair.ver_key(),
                extra_proof_bound_data,
            },
        };

        Ok((transfer_note, signing_keypair))
    }

    /// Anonymous transfer note verification method
    /// * `verifier_key` - Verification key
    /// * `merkle_root` - candidate state of the accumulator. It must match
    ///   note.aux_info.merkle_root, otherwise it returns
    ///   CustomError::TransferVerification Error.
    /// * `timestamp` - current timestamp
    pub fn verify(
        &self,
        verifying_key: &TransferVerifyingKey<C>,
        merkle_root: NodeValue<C::ScalarField>,
        timestamp: u64,
    ) -> Result<(), TxnApiError> {
        // build public inputs
        let pub_inputs =
            self.check_instance_and_get_public_input_internal(merkle_root, timestamp)?;

        // verify proof
        crate::proof::transfer::verify(
            verifying_key,
            &pub_inputs,
            &self.proof,
            &self.aux_info.txn_memo_ver_key,
            &self.aux_info.extra_proof_bound_data,
        )
    }

    /// Anonymous transfer note verification method
    /// * `merkle_root` - candidate state of the accumulator. It must match
    ///   note.aux_info.merkle_root, otherwise it returns
    ///   CustomError::TransferVerification Error.
    /// * `timestamp` - current timestamp
    pub(crate) fn check_instance_and_get_public_input_internal(
        &self,
        merkle_root: NodeValue<C::ScalarField>,
        timestamp: u64,
    ) -> Result<TransferPublicInput<C>, TxnApiError> {
        // check root consistency
        if merkle_root != self.aux_info.merkle_root {
            return Err(TxnApiError::FailedTransactionVerification(
                "Merkle root do not match".to_string(),
            ));
        }

        // check validity timeframe
        if timestamp > self.aux_info.valid_until {
            return Err(TxnApiError::FailedTransactionVerification(format!(
                "Expired proofs, expected the proofs to be valid until at least: {}, but found: {}",
                timestamp, self.aux_info.valid_until
            )));
        }

        // build public inputs
        Ok(TransferPublicInput {
            merkle_root,
            #[cfg(not(feature = "transfer_non_native_fee"))]
            native_asset_code: AssetCode::native(),
            valid_until: self.aux_info.valid_until,
            fee: self.aux_info.fee,
            input_nullifiers: self.inputs_nullifiers.clone(),
            output_commitments: self.output_commitments.clone(),
            viewing_memo: self.viewing_memo.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        constants::ATTRS_LEN,
        errors::TxnApiError,
        keys::UserKeyPair,
        prelude::{CapConfig, Config},
        proof::{
            transfer::{preprocess, TransferProvingKey, TransferVerifyingKey},
            universal_setup_for_staging,
        },
        structs::{Amount, AssetDefinition, ExpirableCredential, NoteType},
        transfer::TransferNote,
        utils::{
            compute_universal_param_size,
            params_builder::{PolicyRevealAttr, TransferParamsBuilder},
        },
        TransactionNote,
    };
    use ark_ff::UniformRand;
    use ark_std::{boxed::Box, vec};
    use jf_primitives::{merkle_tree::NodeValue, signatures::schnorr};

    type F = <Config as CapConfig>::ScalarField;

    #[test]
    fn test_anon_xfr_2in_6out() {
        let depth = 10;
        let num_input = 2;
        let num_output = 6;
        let cred_expiry = 9999;
        let valid_until = 1234;
        let extra_proof_bound_data = "0x12345678901234567890".as_bytes().to_vec();

        let mut prng = ark_std::test_rng();
        let domain_size = compute_universal_param_size::<Config>(
            NoteType::Transfer,
            num_input,
            num_output,
            depth,
        )
        .unwrap();
        let srs = universal_setup_for_staging::<_, Config>(domain_size, &mut prng).unwrap();
        let (prover_key, verifier_key, _) = preprocess(&srs, num_input, num_output, depth).unwrap();

        let keypair1 = UserKeyPair::<Config>::generate(&mut prng);
        let keypair2 = UserKeyPair::<Config>::generate(&mut prng);

        // ====================================
        // a transfer with 0 fee
        // ====================================
        let input_amounts = Amount::from_vec(&[30, 25]);
        let output_amounts = Amount::from_vec(&[30, 3, 4, 5, 6, 7]);

        let mut builder = test_anon_xfr_helper(
            &input_amounts,
            &output_amounts,
            &keypair1,
            &keypair2,
            depth,
            &prover_key,
            &verifier_key,
            valid_until,
            cred_expiry,
            &extra_proof_bound_data,
        )
        .unwrap();

        // ====================================
        // a normal transfer
        // ====================================
        let input_amounts = Amount::from_vec(&[30, 25]);
        let output_amounts = Amount::from_vec(&[19, 3, 4, 5, 6, 7]);

        let _builder = test_anon_xfr_helper(
            &input_amounts,
            &output_amounts,
            &keypair1,
            &keypair2,
            depth,
            &prover_key,
            &verifier_key,
            valid_until,
            cred_expiry,
            &extra_proof_bound_data,
        )
        .unwrap();

        // ====================================
        // bad prover
        // ====================================
        // 1. inconsistent prover_crs
        let mut prover_key = prover_key;
        prover_key.tree_depth += 1;
        assert!(builder
            .build_transfer_note(
                &mut prng,
                &prover_key,
                valid_until,
                extra_proof_bound_data.clone()
            )
            .is_err());
        prover_key.tree_depth -= 1;
        prover_key.n_inputs += 1;
        assert!(builder
            .build_transfer_note(
                &mut prng,
                &prover_key,
                valid_until,
                extra_proof_bound_data.clone()
            )
            .is_err());
        prover_key.n_inputs -= 1;
        prover_key.n_outputs += 1;
        assert!(builder
            .build_transfer_note(
                &mut prng,
                &prover_key,
                valid_until,
                extra_proof_bound_data.clone()
            )
            .is_err());
        prover_key.n_outputs -= 1;

        // 2. empty inputs/outputs
        assert!(TransferNote::generate(
            &mut prng,
            vec![],
            &builder.output_ros,
            &prover_key,
            valid_until,
            vec![]
        )
        .is_err());
        builder.output_ros.truncate(1); // all but fee change
        assert!(builder
            .build_transfer_note(
                &mut prng,
                &prover_key,
                valid_until,
                extra_proof_bound_data.clone()
            )
            .is_err());
        builder.output_ros = vec![]; // prune all output and reset
        let builder =
            builder.set_output_amounts(19u64.into(), &Amount::from_vec(&[3, 4, 5, 6, 7])[..]);

        // 3.invalid inputs/outputs
        let mut builder = builder;
        builder.input_ros[0].amount += Amount::from(1u64);
        assert!(builder
            .build_transfer_note(
                &mut prng,
                &prover_key,
                valid_until,
                extra_proof_bound_data.clone()
            )
            .is_err());
        builder.input_ros[0].amount -= Amount::from(1u64);

        // 4. inconsistent MT roots
        let mut mt_info = builder.input_acc_member_witnesses[0].clone();
        mt_info.root = NodeValue::default();
        builder.input_acc_member_witnesses[0] = mt_info;
        assert!(builder
            .build_transfer_note(
                &mut prng,
                &prover_key,
                valid_until,
                extra_proof_bound_data.clone()
            )
            .is_err());

        // 5. test invalid cred
        assert!(
            builder
                .build_transfer_note(
                    &mut prng,
                    &prover_key,
                    cred_expiry + 1,
                    extra_proof_bound_data.clone()
                )
                .is_err(),
            "expired credential should fail"
        );
        let wrong_cred = ExpirableCredential::dummy_unexpired().unwrap();
        let correct_cred = builder.input_creds[1].clone();
        builder.input_creds[1] = Some(wrong_cred);
        assert!(
            builder
                .build_transfer_note(
                    &mut prng,
                    &prover_key,
                    valid_until,
                    extra_proof_bound_data.clone()
                )
                .is_err(),
            "wrong credential should fail"
        );
        builder.input_creds[1] = correct_cred;

        // 6. Multiple non-native asset type should fail, currently only support
        // single type transfer (apart from native for fee)
        {
            let keypair = UserKeyPair::<Config>::generate(&mut prng);
            let num_input = 3;
            let num_output = 3;

            let domain_size = compute_universal_param_size::<Config>(
                NoteType::Transfer,
                num_input,
                num_output,
                depth,
            )
            .unwrap();
            let srs = universal_setup_for_staging::<_, Config>(domain_size, &mut prng).unwrap();

            let (prover_key, ..) = preprocess(&srs, num_input, num_output, depth).unwrap();

            let second_asset_def = AssetDefinition::rand_for_test(&mut prng);
            let builder = TransferParamsBuilder::new_non_native(
                num_input,
                num_output,
                Some(depth),
                vec![&keypair; num_input],
            )
            .set_input_amounts(
                Amount::from(30u64),
                &[Amount::from(20u64), Amount::from(10u64)],
            )
            .set_output_amounts(
                Amount::from(19u64),
                &[Amount::from(20u64), Amount::from(10u64)],
            )
            .set_input_creds(cred_expiry)
            .update_input_asset_def(1, second_asset_def.clone())
            .update_output_asset_def(1, second_asset_def);
            assert!(
                builder
                    .build_transfer_note(
                        &mut prng,
                        &prover_key,
                        valid_until,
                        extra_proof_bound_data.clone()
                    )
                    .is_err(),
                "Multiple non-native asset type should fail"
            );
        }
    }

    fn test_anon_xfr_helper<'a>(
        input_amounts: &[Amount],
        output_amounts: &[Amount],
        keypair1: &'a UserKeyPair<Config>,
        keypair2: &'a UserKeyPair<Config>,
        depth: u8,
        prover_key: &TransferProvingKey<Config>,
        verifier_key: &TransferVerifyingKey<Config>,
        valid_until: u64,
        cred_expiry: u64,
        extra_proof_bound_data: &[u8],
    ) -> Result<TransferParamsBuilder<'a, Config>, TxnApiError> {
        let mut prng = &mut ark_std::test_rng();

        let builder = TransferParamsBuilder::new_non_native(
            input_amounts.len(),
            output_amounts.len(),
            Some(depth),
            vec![&keypair1, &keypair2],
        )
        .set_input_amounts(input_amounts[0], &input_amounts[1..])
        .set_output_amounts(output_amounts[0], &output_amounts[1..])
        .policy_reveal(PolicyRevealAttr::Amount)
        .set_input_creds(cred_expiry);

        let (note, recv_memos, sig) = builder
            .build_transfer_note(
                &mut prng,
                &prover_key,
                valid_until,
                extra_proof_bound_data.to_vec(),
            )
            .unwrap();

        // Check memos
        let asset_def = builder.transfer_asset_def.as_ref().unwrap();

        let viewer_keypair = &asset_def.viewer_keypair;

        let (input_visible_data, output_visible_data) = viewer_keypair
            .open_transfer_viewing_memo(&asset_def.asset_def, &note)
            .unwrap();
        assert_eq!(input_visible_data.len(), input_amounts.len() - 1);
        assert_eq!(output_visible_data.len(), output_amounts.len() - 1);

        assert_eq!(input_visible_data[0].asset_code, asset_def.asset_def.code);
        assert_eq!(input_visible_data[0].amount, Some(input_amounts[1]));
        assert_eq!(input_visible_data[0].attributes.len(), ATTRS_LEN);
        assert!(input_visible_data[0].blinding_factor.is_none());
        assert!(input_visible_data[0].user_address.is_none());

        for (visible_data, expected_amount) in output_visible_data.iter().zip(&output_amounts[1..])
        {
            assert_eq!(visible_data.asset_code, asset_def.asset_def.code);
            assert_eq!(visible_data.amount, Some(*expected_amount));
            assert_eq!(visible_data.attributes.len(), ATTRS_LEN);
            assert!(visible_data.blinding_factor.is_none());
            assert!(visible_data.user_address.is_none());
        }

        assert!(note
            .verify(&verifier_key, builder.root, valid_until - 1)
            .is_ok());
        assert!(note
            .verify(&verifier_key, builder.root, valid_until)
            .is_ok());
        assert!(note
            .verify(&verifier_key, builder.root, valid_until + 1)
            .is_err());
        assert!(note
            .verify(&verifier_key, NodeValue::default(), valid_until)
            .is_err());
        // note with wrong recv_memos_ver_key should fail
        let mut wrong_note = note.clone();
        wrong_note.aux_info.txn_memo_ver_key = schnorr::KeyPair::generate(&mut prng).ver_key();
        assert!(wrong_note
            .verify(&verifier_key, builder.root, valid_until - 1)
            .is_err());
        // note with wrong `extra_proof_bound_data` should fail
        let mut wrong_note = note.clone();
        wrong_note.aux_info.extra_proof_bound_data = vec![];
        assert!(wrong_note
            .verify(&verifier_key, builder.root, valid_until - 1)
            .is_err());

        // test receiver memos and signature
        assert!(
            TransactionNote::Transfer(Box::new(note))
                .verify_receiver_memos_signature(&recv_memos, &sig)
                .is_ok(),
            "Should have correct receiver memo signature"
        );

        Ok(builder)
    }

    #[test]
    fn xfr_with_dummy_inputs() {
        let depth = 10;
        let num_input = 4;
        let num_output = 6;
        let cred_expiry = 9999;
        let valid_until = 1234;
        let extra_proof_bound_data = "0x12345678901234567890".as_bytes().to_vec();

        let mut prng = ark_std::test_rng();
        let domain_size = compute_universal_param_size::<Config>(
            NoteType::Transfer,
            num_input,
            num_output,
            depth,
        )
        .unwrap();
        let srs = universal_setup_for_staging::<_, Config>(domain_size, &mut prng).unwrap();
        let (prover_key, verifier_key, _) =
            preprocess::<Config>(&srs, num_input, num_output, depth).unwrap();

        let fee_input = Amount::from(30u64);
        let fee_chg = Amount::from(19u64);
        let input_amounts = Amount::from_vec(&[10, 0, 20]);
        let output_amounts = Amount::from_vec(&[2, 3, 4, 5, 16]);

        let keypair1 = UserKeyPair::generate(&mut prng);
        let keypair2 = UserKeyPair::generate(&mut prng);
        let dummy_keypair = Default::default();
        let mut builder = TransferParamsBuilder::new_non_native(
            num_input,
            num_output,
            Some(depth),
            vec![&keypair1, &keypair1, &dummy_keypair, &keypair2],
        )
        .set_input_amounts(fee_input, &input_amounts)
        .set_output_amounts(fee_chg, &output_amounts)
        .policy_reveal(PolicyRevealAttr::Amount)
        .set_input_creds(cred_expiry)
        .update_input_asset_def(1, AssetDefinition::dummy());

        // put garbage on acc_member_witness
        assert_ne!(builder.input_acc_member_witnesses[2].uid, 0);
        builder.input_acc_member_witnesses[2].uid = 0;
        for node in builder.input_acc_member_witnesses[2]
            .merkle_path
            .nodes
            .iter_mut()
        {
            node.sibling1 = NodeValue::from_scalar(F::rand(&mut prng));
            node.sibling2 = NodeValue::from_scalar(F::rand(&mut prng));
        }

        let (note, _recv_memos, _sig) = builder
            .build_transfer_note(
                &mut prng,
                &prover_key,
                valid_until,
                extra_proof_bound_data.clone(),
            )
            .unwrap();
        assert!(note
            .verify(&verifier_key, builder.root, valid_until - 1)
            .is_ok());
    }
}
