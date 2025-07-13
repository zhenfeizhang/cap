// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Configurable Asset Privacy (CAP) library.

// This program is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version. This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details. You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

use crate::{
    circuit::{gadgets_helper::TransactionGadgetsHelper, structs::RecordOpeningVar},
    prelude::CapConfig,
};
use ark_ff::One;
use ark_std::{string::ToString, vec::Vec};
use jf_primitives::circuit::merkle_tree::{AccElemVars, AccMemberWitnessVar, MerkleTreeGadget};
use jf_relation::{errors::CircuitError, Circuit, PlonkCircuit, Variable};

#[derive(Clone, PartialEq, Eq, Debug)]
/// Enum for an asset record spender.
pub(crate) enum Spender {
    User,
    Freezer,
}

// High-level transaction related gadgets
pub(crate) trait TransactionGadgets<C: CapConfig> {
    /// Add constraints that enforces the balance between inputs and outputs.
    /// Return the input transfer amount (which excludes the fee input amount).
    /// Enforces
    ///   * `sum_{i=0..n} amounts_in[i] == sum_{i=0..m} amounts_out[i]`
    ///
    /// The input parameters are:
    /// * `amounts_in` - input amounts, **should be non-empty**
    /// * `amounts_out` - output amounts, **should be non-empty**
    fn preserve_balance(
        &mut self,
        amounts_in: &[Variable],
        amounts_out: &[Variable],
    ) -> Result<Variable, CircuitError>;

    /// Prove the possession of an asset record and spend it,
    /// add the corresponding constraints.
    /// * `ro` - the variables for the asset record opening
    /// * `acc_member_witness` - (uid, merkle path) Merkle proof of record
    /// * `sk` - a secret key variable used to spend the asset
    /// * `spender` - the identity of the spender, can be user or freezer.
    /// * output - (`nullifier`, `root`): nullifier, and the Merkle root value.
    fn prove_spend(
        &mut self,
        ro: &RecordOpeningVar,
        acc_member_witness: &AccMemberWitnessVar,
        sk: Variable,
        spender: Spender,
    ) -> Result<(Variable, Variable), CircuitError>;

    /// Apply hadamard product on `vals` and binary vector `bit_map_vars`.
    fn hadamard_product(
        &mut self,
        bit_map_vars: &[Variable],
        vals: &[Variable],
    ) -> Result<Vec<Variable>, CircuitError>;
}

impl<C: CapConfig> TransactionGadgets<C> for PlonkCircuit<C::ScalarField> {
    fn preserve_balance(
        &mut self,
        amounts_in: &[Variable],
        amounts_out: &[Variable],
    ) -> Result<Variable, CircuitError> {
        // FIXME(ZZ): This code is not sound.
        // We need to constraint all inputs and outputs are non-negative.
        if amounts_in.is_empty() {
            return Err(CircuitError::InternalError(
                "amounts_in is empty".to_string(),
            ));
        }
        if amounts_out.is_empty() {
            return Err(CircuitError::InternalError(
                "amounts_out is empty".to_string(),
            ));
        }
        let total_amounts_in = self.sum(&amounts_in)?;
        let total_amounts_out = self.sum(&amounts_out)?;
        self.enforce_equal(total_amounts_in, total_amounts_out)?;

        Ok(total_amounts_in)
    }

    fn prove_spend(
        &mut self,
        ro: &RecordOpeningVar,
        acc_member_witness: &AccMemberWitnessVar,
        sk: Variable,
        spender: Spender,
    ) -> Result<(Variable, Variable), CircuitError> {
        let (uid, path_ref) = (acc_member_witness.uid, &acc_member_witness.merkle_path);
        let (pk1_point, pk2_point) = if spender == Spender::User {
            (&ro.owner_addr.0, &ro.policy.freezer_pk)
        } else {
            (&ro.policy.freezer_pk, &ro.owner_addr.0)
        };

        // PoK of secret key
        let pk = TransactionGadgetsHelper::<C>::derive_user_address(self, sk)?;
        self.enforce_point_equal(&pk.0, pk1_point)?;

        // compute commitment
        let commitment = ro.compute_record_commitment::<C>(self)?;

        // derive nullify key and compute nullifier
        let nk = TransactionGadgetsHelper::<C>::derive_nullifier_key(self, sk, pk2_point)?;
        let nullifier = TransactionGadgetsHelper::<C>::nullify(self, nk, uid, commitment)?;

        // verify Merkle path
        let root = self.compute_merkle_root(
            AccElemVars {
                uid,
                elem: commitment,
            },
            path_ref,
        )?;

        Ok((nullifier, root))
    }

    fn hadamard_product(
        &mut self,
        bit_map_vars: &[Variable],
        vals: &[Variable],
    ) -> Result<Vec<Variable>, CircuitError> {
        if bit_map_vars.len() != vals.len() {
            return Err(CircuitError::InternalError(
                "expecting the same length for vals and reveal_map".to_string(),
            ));
        }
        bit_map_vars
            .iter()
            .zip(vals.iter())
            .map(|(&bit, &val)| self.mul(bit, val))
            .collect::<Result<Vec<_>, CircuitError>>()
    }
}

#[cfg(test)]
mod tests {
    use super::{Spender, TransactionGadgets};
    use crate::{
        circuit::structs::RecordOpeningVar,
        constants::VIEWABLE_DATA_LEN,
        keys::{FreezerKeyPair, FreezerPubKey, UserKeyPair},
        prelude::{CapConfig, Config},
        structs::{AssetPolicy, RecordCommitment, RecordOpening, RevealMap},
    };
    use ark_ff::{One, Zero};
    use ark_std::UniformRand;
    use ark_std::{test_rng, vec::Vec};
    use jf_primitives::{
        circuit::merkle_tree::{gen_merkle_path_for_test, AccMemberWitnessVar},
        merkle_tree::AccMemberWitness,
    };
    use jf_relation::{errors::CircuitError, Circuit, PlonkCircuit, Variable};
    use jf_utils::fr_to_fq;
    use rand::{Rng, RngCore};

    type F = <Config as CapConfig>::ScalarField;
    type EmbeddedCurveParam = <Config as CapConfig>::EmbeddedCurveParam;

    fn build_preserve_balance_circuit(
        amounts_in: &[F],
        amounts_out: &[F],
    ) -> Result<PlonkCircuit<F>, CircuitError> {
        let expected_transfer_amount = amounts_in.iter().fold(F::zero(), |acc, &x| acc + x);
        let mut circuit = PlonkCircuit::new_turbo_plonk();
        let amounts_in: Vec<Variable> = amounts_in
            .iter()
            .map(|&val| circuit.create_variable(val))
            .collect::<Result<Vec<_>, CircuitError>>()?;
        let amounts_out: Vec<Variable> = amounts_out
            .iter()
            .map(|&val| circuit.create_variable(val))
            .collect::<Result<Vec<_>, CircuitError>>()?;
        let transfer_amount = TransactionGadgets::<Config>::preserve_balance(
            &mut circuit,
            &amounts_in,
            &amounts_out,
        )?;

        assert_eq!(expected_transfer_amount, circuit.witness(transfer_amount)?);
        Ok(circuit)
    }

    fn sample_amounts(
        rng: &mut impl RngCore,
        num_inputs: usize,
        num_outputs: usize,
        is_sound: bool,
    ) -> (Vec<F>, Vec<F>) {
        let amounts_in: Vec<F> = (0..num_inputs)
            .map(|_| rng.gen_range(0..256).into())
            .collect();
        let mut amounts_out: Vec<F> = (0..num_outputs - 1)
            .map(|_| rng.gen_range(0..256).into())
            .collect();
        let total_in: F = amounts_in.iter().sum();
        let total_out: F = amounts_out.iter().sum();

        amounts_out.push(total_in - total_out);

        if !is_sound {
            amounts_out[0] = amounts_out[0] + F::one(); // make it unsound
        }

        (amounts_in, amounts_out)
    }

    #[test]
    fn test_preserve_balance() -> Result<(), CircuitError> {
        let rng = &mut test_rng();

        for num_in in [1, 2, 4] {
            for num_out in [1, 2, 4] {
                for is_sound in [true, false] {
                    let (amounts_in, amounts_out) = sample_amounts(rng, num_in, num_out, is_sound);
                    let circuit = build_preserve_balance_circuit(&amounts_in, &amounts_out)?;
                    if is_sound {
                        assert!(circuit.check_circuit_satisfiability(&[]).is_ok());
                    } else {
                        assert!(circuit.check_circuit_satisfiability(&[]).is_err());
                    }
                }
            }
        }

        Ok(())
    }

    fn check_prove_spend_circuit(
        ro: &RecordOpening<Config>,
        acc_member_witness: &AccMemberWitness<F>,
        sk: F,
        spender: Spender,
        expected_nullifier: F,
        expected_root: F,
    ) -> Result<(), CircuitError> {
        let mut circuit = PlonkCircuit::<F>::new_turbo_plonk();
        let ro_var = RecordOpeningVar::new(&mut circuit, ro)?;
        let acc_wit_var =
            AccMemberWitnessVar::new::<_, EmbeddedCurveParam>(&mut circuit, &acc_member_witness)?;

        let sk_var = circuit.create_variable(sk)?;
        let (nullifier, root) = TransactionGadgets::<Config>::prove_spend(
            &mut circuit,
            &ro_var,
            &acc_wit_var,
            sk_var,
            spender,
        )?;

        assert_eq!(circuit.witness(nullifier)?, expected_nullifier);
        assert_eq!(circuit.witness(root)?, expected_root);
        assert!(circuit.check_circuit_satisfiability(&[]).is_ok());
        *circuit.witness_mut(root) = F::one();
        assert!(circuit.check_circuit_satisfiability(&[]).is_err());

        Ok(())
    }

    #[test]
    fn test_prove_spend() -> Result<(), CircuitError> {
        let rng = &mut test_rng();

        // Case 1: Asset record with freezing policy
        // Create user/freezer keypairs
        let user_keypair = UserKeyPair::generate(rng);
        let freezer_keypair = FreezerKeyPair::generate(rng);
        // Create user's asset record
        let mut ro = RecordOpening::rand_for_test(rng);
        ro.asset_def.policy.freezer_pk = freezer_keypair.pub_key();
        ro.pub_key = user_keypair.pub_key();
        // Compute expected nullifier and root
        let ro_comm = ro.derive_record_commitment();
        let uid = 2u64;
        let expected_nl = freezer_keypair.nullify(&user_keypair.address(), uid, &ro_comm);
        let (acc_wit, expected_root) = gen_merkle_path_for_test(uid, ro_comm.0);
        // Check user spending
        let usk = fr_to_fq::<_, EmbeddedCurveParam>(user_keypair.address_secret_ref());
        check_prove_spend_circuit(
            &ro,
            &acc_wit,
            usk,
            Spender::User,
            expected_nl.0,
            expected_root,
        )?;
        // Check freezer spending
        let fsk = fr_to_fq::<_, EmbeddedCurveParam>(&freezer_keypair.sec_key);
        check_prove_spend_circuit(
            &ro,
            &acc_wit,
            fsk,
            Spender::Freezer,
            expected_nl.0,
            expected_root,
        )?;

        // Case 2: Asset record with no freezing policy.
        let mut ro = RecordOpening::rand_for_test(rng);
        ro.asset_def.policy = AssetPolicy::default();
        ro.pub_key = user_keypair.pub_key();
        let ro_comm = RecordCommitment::from(&ro);
        let uid = 3u64;
        let expected_nl = user_keypair.nullify(&FreezerPubKey::default(), uid, &ro_comm);
        let (acc_wit, expected_root) = gen_merkle_path_for_test(uid, ro_comm.0);
        check_prove_spend_circuit(
            &ro,
            &acc_wit,
            usk,
            Spender::User,
            expected_nl.0,
            expected_root,
        )?;

        Ok(())
    }

    fn check_hadamard_product<C: CapConfig<ScalarField = F>>(
        reveal_map: &RevealMap,
        vals: &[F],
        bit_len: usize,
    ) -> Result<(), CircuitError> {
        let expected_hadamard = reveal_map.hadamard_product::<C>(vals);
        let mut circuit = PlonkCircuit::new_turbo_plonk();
        let reveal_map_var = circuit.create_variable(reveal_map.to_scalar::<C>())?;
        let bit_map_vars: Vec<Variable> = circuit
            .unpack(reveal_map_var, VIEWABLE_DATA_LEN)?
            .into_iter()
            .rev()
            .map(|bv| bv.into())
            .collect();
        let vals = vals
            .iter()
            .map(|&val| circuit.create_variable(val))
            .collect::<Result<Vec<_>, CircuitError>>()?;
        let prod = TransactionGadgets::<Config>::hadamard_product(
            &mut circuit,
            &bit_map_vars[..bit_len],
            &vals[..bit_len],
        )?;

        for i in 0..bit_len {
            assert_eq!(circuit.witness(prod[i])?, expected_hadamard[i]);
        }
        assert!(circuit.check_circuit_satisfiability(&[]).is_ok());
        *circuit.witness_mut(prod[0]) = F::one();
        assert!(circuit.check_circuit_satisfiability(&[]).is_err());

        Ok(())
    }

    #[test]
    fn test_hadamard_product() -> Result<(), CircuitError> {
        let mut reveal_map = RevealMap::default();
        reveal_map.reveal_all();
        let vals: Vec<F> = (0..VIEWABLE_DATA_LEN).map(|i| F::from(i as u32)).collect();
        check_hadamard_product::<Config>(&reveal_map, &vals, VIEWABLE_DATA_LEN)?;

        let reveal_map = RevealMap::default();
        check_hadamard_product::<Config>(&reveal_map, &vals, VIEWABLE_DATA_LEN)?;

        let rng = &mut ark_std::test_rng();
        let reveal_map = RevealMap::rand_for_test(rng);
        check_hadamard_product::<Config>(&reveal_map, &vals, VIEWABLE_DATA_LEN)?;

        check_hadamard_product::<Config>(&reveal_map, &vals, 4)?;
        Ok(())
    }
}
