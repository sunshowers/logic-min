// Copyright (c) The logic-min Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::cube::{Cube, CubeSet};

impl<const IL: usize, const OL: usize> CubeSet<IL, OL> {
    /// Given a cube, generates prime implicants for that cube.
    pub fn prime_implicants(self) -> Self {
        println!("computing prime implicants for: {:?}", self);
        // If there's just 0 or 1 product terms, stop.
        if self.cube_count() <= 1 {
            return self;
        }
        // If this is a tautology, return the tautological set (total universe).
        if self.is_tautology() {
            println!("*** is tautology");
            return CubeSet::new([Cube::total_universe()]);
        }

        match self.make_unate_or_select_binate() {
            Ok(unate_set) => {
                // The prime implicants are simply unate_set.simplify().
                unate_set.simplify().into_inner()
            }
            Err((cube_set, max_binate_ix)) => {
                let mut expansion = cube_set.shannon_expansion(max_binate_ix);
                println!("max binate ix: {}", max_binate_ix);
                expansion.transform(|cube_set| {
                    println!("computing prime implicants for half-set: {:?}", cube_set);
                    cube_set.prime_implicants()
                });

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
    use crate::cube::CubeSet;

    #[test]
    fn test_basic() {
        let cube_set =
            CubeSet::from_numeric0([[0, 2, 1], [0, 1, 2], [0, 1, 1], [1, 2, 1], [1, 2, 0]])
                .unwrap();
        println!("{:?}", cube_set.prime_implicants());
    }
}
