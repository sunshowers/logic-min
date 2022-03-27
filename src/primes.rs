// Copyright (c) The logic-min Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::cover::Cover;
use crate::cube::Cube;

impl<const IL: usize, const OL: usize> Cover<IL, OL> {
    /// Given a cube, generates prime implicants for that cube.
    pub fn prime_implicants(self) -> Self {
        self.prime_implicants_impl()
    }

    fn prime_implicants_impl(self) -> Self {
        // Terminal case: if there's just 0 or 1 cubes.
        if self.cube_count() <= 1 {
            return self;
        }

        // TODO: switch this to single input dependence which apparently is enough.
        if self.is_tautology() {
            return Cover::new([Cube::total_universe()]);
        }

        match self.make_unate_or_select_binate() {
            Ok(unate_set) => {
                // The prime implicants are simply unate_set.simplify().
                unate_set.simplify().into_inner()
            }
            Err((cover, max_binate_ix)) => {
                let mut expansion = cover.shannon_expansion(max_binate_ix);
                expansion.transform(|cover| cover.prime_implicants_impl());

                let pos_result = expansion.pos_half_space() & expansion.positive();
                let neg_result = expansion.neg_half_space() & expansion.negative();

                let consensus = pos_result.consensus(&neg_result);

                let union_set = pos_result | neg_result | consensus;
                union_set.single_cube_containment()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let cover =
            Cover::from_numeric0([[0, 2, 1], [0, 1, 2], [0, 1, 1], [1, 2, 1], [1, 2, 0]]).unwrap();
        assert_eq!(
            cover.prime_implicants(),
            Cover::from_numeric0([[1, 2, 2], [2, 1, 2], [2, 2, 1]]).unwrap()
        );
    }
}
