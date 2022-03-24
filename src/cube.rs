// Copyright (c) The logic-min Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::errors::InvalidCubeNumeric;
use arrayvec::ArrayVec;
use smallvec::SmallVec;
use std::ops::BitAnd;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cube<const IL: usize, const OL: usize> {
    pub input: [Option<bool>; IL],
    pub output: [bool; OL],
}

impl<const IL: usize, const OL: usize> Cube<IL, OL> {
    pub const INPUT_LEN: usize = IL;
    pub const OUTPUT_LEN: usize = OL;

    // Uses the representation in the Espresso book.
    pub fn from_numeric(
        input_numeric: [u8; IL],
        output_numeric: [u8; OL],
    ) -> Result<Self, InvalidCubeNumeric> {
        let mut input: ArrayVec<Option<bool>, IL> = ArrayVec::new();
        let mut output: ArrayVec<bool, OL> = ArrayVec::new();

        for val in input_numeric {
            match val {
                0 => input.push(Some(false)),
                1 => input.push(Some(true)),
                2 => input.push(None),
                _ => {
                    // IL should be 0, 1 or 2
                    return Err(InvalidCubeNumeric);
                }
            }
        }

        for val in output_numeric {
            match val {
                3 => output.push(false),
                4 => output.push(true),
                _ => {
                    // OL should be 3 or 4
                    return Err(InvalidCubeNumeric);
                }
            }
        }

        // SAFETY: we push exactly as many as IL or OL
        let input = unsafe { input.into_inner_unchecked() };
        let output = unsafe { output.into_inner_unchecked() };

        Ok(Cube { input, output })
    }

    pub fn universe(output_index: usize) -> Self {
        let input = [None; IL];
        let mut output = [false; OL];
        output[output_index] = true;
        Self { input, output }
    }

    pub fn total_universe() -> Self {
        let input = [None; IL];
        let output = [true; OL];

        Self { input, output }
    }

    pub fn positive_half_space(input_index: usize) -> Self {
        let mut input = [None; IL];
        input[input_index] = Some(true);
        let output = [true; OL];

        Self { input, output }
    }

    pub fn contains(&self, other: &Cube<IL, OL>) -> bool {
        self.input
            .iter()
            .zip(&other.input)
            .all(|(&c, &d)| CubeContains::input_contains(c, d) >= CubeContains::Contains)
    }

    pub fn strictly_contains(&self, other: &Cube<IL, OL>) -> bool {
        let mut any_strictly = false;
        let all_contains =
            self.input
                .iter()
                .zip(&other.input)
                .all(|(&c, &d)| match CubeContains::input_contains(c, d) {
                    CubeContains::Strictly => {
                        any_strictly = true;
                        true
                    }
                    CubeContains::Contains => true,
                    CubeContains::DoesNotContain => false,
                });
        all_contains && any_strictly
    }

    pub fn is_minterm(&self, output_index: usize) -> bool {
        // TODO: check that output_index is between 0 and usize
        let input_minterm = self.input.iter().all(|&c| c.is_some());
        let output_minterm = {
            self.output
                .iter()
                .enumerate()
                .all(|(idx, &c)| if idx == output_index { c } else { !c })
        };
        input_minterm && output_minterm
    }

    pub fn input_distance(&self, other: &Cube<IL, OL>) -> usize {
        // page 25
        self.input
            .iter()
            .zip(&other.input)
            .filter(|(&c, &d)| intersect_input_one(c, d).is_phi())
            .count()
    }

    pub fn output_distance(&self, other: &Cube<IL, OL>) -> usize {
        // page 25
        if self
            .output
            .iter()
            .zip(&other.output)
            .any(|(&c, &d)| intersect_output(c, d))
        {
            0
        } else {
            1
        }
    }

    pub fn distance(&self, other: &Cube<IL, OL>) -> usize {
        // page 25
        self.input_distance(other) + self.output_distance(other)
    }

    pub fn consensus(&self, other: &Cube<IL, OL>) -> Option<Self> {
        // page 25
        match (self.input_distance(other), self.output_distance(other)) {
            (0, 0) => self & other,
            (1, 0) => {
                let mut input: ArrayVec<Option<bool>, IL> = ArrayVec::new();
                let mut output: ArrayVec<bool, OL> = ArrayVec::new();

                for (&c, &d) in self.input.iter().zip(&other.input) {
                    match intersect_input_one(c, d) {
                        IntersectInputResult::Value(x) => input.push(x),
                        // "the conflicting input part is raised to 2"
                        IntersectInputResult::Phi => input.push(None),
                    }
                }

                for (&c, &d) in self.output.iter().zip(&other.output) {
                    output.push(intersect_output(c, d));
                }

                // SAFETY: we push exactly as many as IL or OL
                debug_assert_eq!(input.len(), input.capacity());
                let input = unsafe { input.into_inner_unchecked() };
                debug_assert_eq!(output.len(), output.capacity());
                let output = unsafe { output.into_inner_unchecked() };

                Some(Cube { input, output })
            }
            (0, 1) => {
                let mut input: ArrayVec<Option<bool>, IL> = ArrayVec::new();
                let mut output: ArrayVec<bool, OL> = ArrayVec::new();

                for (&c, &d) in self.input.iter().zip(&other.input) {
                    match intersect_input_one(c, d) {
                        IntersectInputResult::Value(x) => input.push(x),
                        IntersectInputResult::Phi => {
                            panic!("input distance 0 means no values are phi")
                        }
                    }
                }

                for (&c, &d) in self.output.iter().zip(&other.output) {
                    output.push(c || d);
                }

                // SAFETY: we push exactly as many as IL or OL
                debug_assert_eq!(input.len(), input.capacity());
                let input = unsafe { input.into_inner_unchecked() };
                debug_assert_eq!(output.len(), output.capacity());
                let output = unsafe { output.into_inner_unchecked() };

                Some(Cube { input, output })
            }
            _ => None,
        }
    }
}

impl<const IL: usize, const OL: usize> BitAnd for Cube<IL, OL> {
    type Output = Option<Cube<IL, OL>>;

    fn bitand(self, rhs: Self) -> Self::Output {
        bitand_impl(&self, &rhs)
    }
}

impl<'a, const IL: usize, const OL: usize> BitAnd<&'a Cube<IL, OL>> for Cube<IL, OL> {
    type Output = Option<Cube<IL, OL>>;

    fn bitand(self, rhs: &'a Cube<IL, OL>) -> Self::Output {
        bitand_impl(&self, rhs)
    }
}

impl<'a, 'b, const IL: usize, const OL: usize> BitAnd<&'a Cube<IL, OL>> for &'b Cube<IL, OL> {
    type Output = Option<Cube<IL, OL>>;

    fn bitand(self, rhs: &'a Cube<IL, OL>) -> Self::Output {
        bitand_impl(self, rhs)
    }
}

fn bitand_impl<const IL: usize, const OL: usize>(
    a: &Cube<IL, OL>,
    b: &Cube<IL, OL>,
) -> Option<Cube<IL, OL>> {
    // page 24

    let mut input: ArrayVec<Option<bool>, IL> = ArrayVec::new();
    let mut output: ArrayVec<bool, OL> = ArrayVec::new();

    for (&c, &d) in a.input.iter().zip(&b.input) {
        match intersect_input_one(c, d) {
            IntersectInputResult::Value(x) => input.push(x),
            IntersectInputResult::Phi => return None,
        }
    }

    let mut any_output_true = false;
    for (&c, &d) in a.output.iter().zip(&b.output) {
        let elem = intersect_output(c, d);
        if elem {
            any_output_true = true;
        }
        output.push(elem);
    }

    if !any_output_true {
        return None;
    }

    // SAFETY: we push exactly as many as IL or OL
    debug_assert_eq!(input.len(), input.capacity());
    let input = unsafe { input.into_inner_unchecked() };
    debug_assert_eq!(output.len(), output.capacity());
    let output = unsafe { output.into_inner_unchecked() };

    Some(Cube { input, output })
}

// Intersect one input bit.
fn intersect_input_one(c: Option<bool>, d: Option<bool>) -> IntersectInputResult {
    match (c, d) {
        (None, d) => IntersectInputResult::Value(d),
        (d, None) => IntersectInputResult::Value(d),
        (Some(false), Some(false)) => IntersectInputResult::Value(Some(false)),
        (Some(true), Some(true)) => IntersectInputResult::Value(Some(true)),
        (Some(false), Some(true)) | (Some(true), Some(false)) => IntersectInputResult::Phi,
    }
}

#[derive(Clone, Copy, Debug)]
enum IntersectInputResult {
    Phi,
    Value(Option<bool>),
}

impl IntersectInputResult {
    fn is_phi(self) -> bool {
        matches!(self, IntersectInputResult::Phi)
    }
}

#[inline]
fn intersect_output(c: bool, d: bool) -> bool {
    c && d
}

// TODO: union?

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum CubeContains {
    DoesNotContain,
    Contains,
    Strictly,
}

impl CubeContains {
    fn input_contains(c: Option<bool>, d: Option<bool>) -> Self {
        // page 23
        match (c, d) {
            (Some(false), Some(false)) => Self::Contains,
            (Some(false), None | Some(true)) => Self::DoesNotContain,
            (Some(true), Some(false) | None) => Self::Contains,
            (Some(true), Some(true)) => Self::Contains,
            (None, Some(true) | Some(false)) => Self::Strictly,
            (None, None) => Self::Contains,
        }
    }
}

pub struct CubeSet<const IL: usize, const OL: usize> {
    elements: SmallVec<[Cube<IL, OL>; 4]>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minterm() {
        // example on page 23
        let minterm = Cube::from_numeric([1, 1, 1], [4, 3]).unwrap();
        let non_minterm = Cube::from_numeric([2, 2, 1], [4, 4]).unwrap();
        assert!(minterm.is_minterm(0));
        assert!(!minterm.is_minterm(1));

        assert!(!non_minterm.is_minterm(0));
        assert!(!non_minterm.is_minterm(1));

        assert!(non_minterm.contains(&minterm));
        assert!(non_minterm.strictly_contains(&minterm));
    }

    #[test]
    fn test_universe() {
        let universe = Cube::<3, 2>::universe(1);
        let total_universe = Cube::<3, 2>::total_universe();

        assert!(total_universe.contains(&universe));
    }
}
