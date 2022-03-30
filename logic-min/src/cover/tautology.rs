// Copyright (c) The logic-min Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{cover::Cover, cube::Cube};

impl<const IL: usize, const OL: usize> Cover<IL, OL> {
    pub fn is_tautology(&self) -> bool {
        // The empty cover is not a tautology.
        let elements = self.elements();

        if elements.is_empty() {
            return false;
        }

        // Scan through the cover looking for rows where all the inputs are Nones. If all the
        // outputs are covered by such rows, this is a tautology.
        if OL == 0 {
            if elements.contains(&Cube::<IL, OL>::total_universe()) {
                return true;
            }
        } else {
            let mut output_bits = [false; OL];
            for cube in elements {
                if cube.input.iter().all(|i| i.is_none()) {
                    for output_ix in 0..OL {
                        output_bits[output_ix] |= cube.output[output_ix];
                    }
                }
            }
            if output_bits.iter().all(|bit| *bit) {
                return true;
            }
        }

        let meaningful_count = self.meaningful_input_count();
        if meaningful_count <= 7 {
            let mut values = [false; 7];
            for input_bits in 0..2_u8.pow(meaningful_count as u32) {
                for bit in 0..meaningful_count {
                    values[bit] = (input_bits >> bit) & 1 == 1;
                }
                if let Some(cover0) = self.try_as_cover0() {
                    if !cover0.evaluate0_meaningful(&values[0..meaningful_count]) {
                        return false;
                    }
                } else {
                    if self
                        .evaluate_meaningful(&values[0..meaningful_count])
                        .iter()
                        .any(|x| !*x)
                    {
                        return false;
                    }
                }
            }
            return true;
        }

        // TODO: unate reduction/component reduction
        panic!("unimplemented")
    }

    #[cfg(test)]
    fn is_tautology_exhaustive(&self) -> bool {
        match self.try_as_cover0() {
            Some(cover0) => cover0.is_tautology0_exhaustive(),
            None => {
                // Check that each output component is tautological.
                for output_ix in 0..OL {
                    let component = Cover::new(self.output_component(output_ix).cloned());
                    if !component.is_tautology0_exhaustive() {
                        return false;
                    }
                }
                true
            }
        }
    }
}

impl<const IL: usize> Cover<IL, 0> {
    pub fn is_tautology0(&self) -> bool {
        // The empty cube is not a tautology.
        if self.is_empty() {
            return false;
        }

        let elements = self.elements();

        // The tautological cube is present in this cover.
        if elements.contains(&Cube::total_universe()) {
            return true;
        }

        // There's a column of all 1s or all 0s.
        'outer: for input_ix in 0..IL {
            let mut any_0s = false;
            let mut any_1s = false;
            for elem in elements {
                match elem.input[input_ix] {
                    Some(true) => {
                        any_1s = true;
                    }
                    Some(false) => {
                        any_0s = true;
                    }
                    None => {
                        // There's a don't care in this column.
                        continue 'outer;
                    }
                }
            }

            // The cover should have both 0s and 1s.
            if !(any_0s && any_1s) {
                return false;
            }
        }

        // TODO: deficient vertex count.

        // TODO: splitting and reduction -- this is just a truth table search for now.
        self.is_tautology0_exhaustive()
    }

    fn is_tautology0_exhaustive(&self) -> bool {
        for input_bits in 0..2_u32.pow(IL as u32) {
            let mut values = [false; IL];
            for bit in 0..IL {
                if (input_bits >> bit) & 1 == 1 {
                    values[bit] = true;
                }
            }
            if !self.evaluate0(&values) {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_tautology_basic() {
        let all_twos = Cover::from_numeric0([[2, 2, 2]]).unwrap();
        assert!(all_twos.is_tautology(), "all twos is tautological");

        let all_twos_multi = Cover::from_numeric([([2, 2, 2], [4, 4])]).unwrap();
        assert!(
            all_twos_multi.is_tautology(),
            "all twos multi is tautological"
        );

        let all_twos_multi2 =
            Cover::from_numeric([([2, 2, 2], [4, 3]), ([2, 2, 2], [3, 4])]).unwrap();
        assert!(
            all_twos_multi2.is_tautology(),
            "all twos multi2 is tautological"
        );

        let all_zeroes_column = Cover::from_numeric0([[0, 1, 2], [0, 2, 1]]).unwrap();
        assert!(
            !all_zeroes_column.is_tautology(),
            "column with all zeroes is not tautological"
        );

        let all_ones_column = Cover::from_numeric0([[2, 1, 1], [1, 2, 1]]).unwrap();
        assert!(
            !all_ones_column.is_tautology(),
            "column with all ones is not tautological"
        );

        let all_zeroes_column_multi =
            Cover::from_numeric([([0, 1, 2], [4, 4]), ([0, 2, 1], [4, 4])]).unwrap();
        assert!(
            !all_zeroes_column_multi.is_tautology(),
            "column with all zeroes (multi) is not tautological"
        );

        let all_zeroes_column_multi2 =
            Cover::from_numeric([([2, 1, 0], [4, 3]), ([1, 2, 0], [3, 4])]).unwrap();
        assert!(
            !all_zeroes_column_multi2.is_tautology(),
            "column with all zeroes (multi2) is not tautological"
        );

        let single_column_input_dependence =
            Cover::from_numeric0([[2, 2, 0, 2], [2, 2, 1, 2]]).unwrap();
        assert!(
            single_column_input_dependence.is_tautology(),
            "single column input dependence is a tautology"
        );
    }

    proptest! {
        #[test]
        fn proptest_tautology(cover: Cover<8, 4>) {
            prop_assert_eq!(
                cover.is_tautology(),
                cover.is_tautology_exhaustive(),
                "tautology matches exhaustive search",
            );
        }
    }
}
