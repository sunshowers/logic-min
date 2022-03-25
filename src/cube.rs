// Copyright (c) The logic-min Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::errors::InvalidCubeNumeric;
use arrayvec::ArrayVec;
use itertools::Itertools;
use std::{
    collections::BTreeSet,
    ops::{BitAnd, BitOr, Mul, Not},
};

#[derive(Clone, Debug, PartialEq, Eq, Ord, PartialOrd)]
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

    pub fn is_implicant_of(&self, cube_set: &CubeSet<IL, OL>) -> bool {
        (self & cube_set).is_empty()
    }

    pub fn cofactor(&self, p: &Self) -> Option<Self> {
        // page 30
        let mut input: ArrayVec<Option<bool>, IL> = ArrayVec::new();
        let mut output: ArrayVec<bool, OL> = ArrayVec::new();

        for (&self_k, &p_k) in self.input.iter().zip(&p.input) {
            if intersect_input_one(self_k, p_k).is_phi() {
                return None;
            }
            match p_k {
                Some(_) => input.push(None),
                None => input.push(self_k),
            }
        }

        for (&self_k, &p_k) in self.output.iter().zip(&p.output) {
            output.push((!p_k) || self_k);
        }

        // SAFETY: we push exactly as many as IL or OL
        debug_assert_eq!(input.len(), input.capacity());
        let input = unsafe { input.into_inner_unchecked() };
        debug_assert_eq!(output.len(), output.capacity());
        let output = unsafe { output.into_inner_unchecked() };

        Some(Self { input, output })
    }
}

/// Complement operation.
impl<const IL: usize, const OL: usize> Not for Cube<IL, OL> {
    type Output = Cube<IL, OL>;

    fn not(self) -> Self::Output {
        self.complement_impl()
    }
}

impl<'a, const IL: usize, const OL: usize> Not for &'a Cube<IL, OL> {
    type Output = Cube<IL, OL>;

    fn not(self) -> Self::Output {
        self.complement_impl()
    }
}

impl<const IL: usize, const OL: usize> Cube<IL, OL> {
    fn complement_impl(&self) -> Self {
        // Only the input is complemented.
        let input: ArrayVec<Option<bool>, IL> = self
            .input
            .iter()
            .map(|&c| match c {
                Some(x) => Some(!x),
                None => None,
            })
            .collect();

        debug_assert_eq!(input.len(), input.capacity());
        let input = unsafe { input.into_inner_unchecked() };
        Self {
            input,
            output: self.output,
        }
    }
}

/// Intersection operation.
impl<const IL: usize, const OL: usize> BitAnd for Cube<IL, OL> {
    type Output = Option<Cube<IL, OL>>;

    fn bitand(self, rhs: Self) -> Self::Output {
        intersection_impl(&self, &rhs)
    }
}

impl<'a, const IL: usize, const OL: usize> BitAnd<&'a Cube<IL, OL>> for Cube<IL, OL> {
    type Output = Option<Cube<IL, OL>>;

    fn bitand(self, rhs: &'a Cube<IL, OL>) -> Self::Output {
        intersection_impl(&self, rhs)
    }
}

impl<'a, 'b, const IL: usize, const OL: usize> BitAnd<&'a Cube<IL, OL>> for &'b Cube<IL, OL> {
    type Output = Option<Cube<IL, OL>>;

    fn bitand(self, rhs: &'a Cube<IL, OL>) -> Self::Output {
        intersection_impl(self, rhs)
    }
}

impl<'a, const IL: usize, const OL: usize> BitAnd<&'a CubeSet<IL, OL>> for Cube<IL, OL> {
    type Output = CubeSet<IL, OL>;

    fn bitand(self, rhs: &'a CubeSet<IL, OL>) -> Self::Output {
        self.intersect_cube_set_impl(rhs)
    }
}

impl<'a, 'b, const IL: usize, const OL: usize> BitAnd<&'a CubeSet<IL, OL>> for &'b Cube<IL, OL> {
    type Output = CubeSet<IL, OL>;

    fn bitand(self, rhs: &'a CubeSet<IL, OL>) -> Self::Output {
        self.intersect_cube_set_impl(rhs)
    }
}

impl<const IL: usize, const OL: usize> BitAnd<CubeSet<IL, OL>> for Cube<IL, OL> {
    type Output = CubeSet<IL, OL>;

    fn bitand(self, rhs: CubeSet<IL, OL>) -> Self::Output {
        self.intersect_cube_set_impl(&rhs)
    }
}

impl<'b, const IL: usize, const OL: usize> BitAnd<CubeSet<IL, OL>> for &'b Cube<IL, OL> {
    type Output = CubeSet<IL, OL>;

    fn bitand(self, rhs: CubeSet<IL, OL>) -> Self::Output {
        self.intersect_cube_set_impl(&rhs)
    }
}

impl<const IL: usize, const OL: usize> Cube<IL, OL> {
    fn intersect_cube_set_impl(&self, cube_set: &CubeSet<IL, OL>) -> CubeSet<IL, OL> {
        CubeSet::new(cube_set.elements.iter().filter_map(|c| self & c))
    }
}

fn intersection_impl<const IL: usize, const OL: usize>(
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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CubeSet<const IL: usize, const OL: usize> {
    pub elements: BTreeSet<Cube<IL, OL>>,
}

impl<const IL: usize, const OL: usize> CubeSet<IL, OL> {
    pub fn new(elements: impl IntoIterator<Item = Cube<IL, OL>>) -> Self {
        Self {
            elements: elements.into_iter().collect(),
        }
    }

    pub fn from_numeric(
        numeric: impl IntoIterator<Item = ([u8; IL], [u8; OL])>,
    ) -> Result<Self, InvalidCubeNumeric> {
        let elements = numeric
            .into_iter()
            .map(|(input, output)| Cube::from_numeric(input, output))
            .collect::<Result<_, _>>()?;
        Ok(Self { elements })
    }

    pub fn cofactor(&self, p: &Cube<IL, OL>) -> Self {
        let elements = self
            .elements
            .iter()
            .filter_map(|elem| elem.cofactor(p))
            .collect();
        Self { elements }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    pub fn is_cover(&self, c: &Cube<IL, OL>) -> bool {
        unimplemented!("need to implement minterms first")
    }

    pub fn shannon_expansion(&self, p: &Cube<IL, OL>) -> Self {
        (p & self.cofactor(p)) | (!p & self.cofactor(&!p))
    }

    /// Given two subcovers self and other, obtained by the Shannon expansion with
    /// respect to a cube p, computes a cover by merging them.
    pub fn merge_with_identity(mut self, mut other: Self, p: &Cube<IL, OL>) -> Self {
        let result = self.intersect_and_remove(&mut other);
        ((!p) & &self) | (p & &other) | result
    }

    pub fn merge_with_containment(mut self, mut other: Self, _p: &Cube<IL, OL>) -> Self {
        let _result = self.intersect_and_remove(&mut other);
        unimplemented!("still need to implement containment")
    }

    pub fn is_monotone_increasing(&self, input_ix: usize) -> bool {
        assert!(
            input_ix < IL,
            "input elem {} must be in range [0..{})",
            input_ix,
            IL
        );
        self.elements
            .iter()
            .all(|elem| elem.input[input_ix] != Some(false))
    }

    pub fn is_monotone_decreasing(&self, input_ix: usize) -> bool {
        assert!(
            input_ix < IL,
            "input elem {} must be in range [0..{})",
            input_ix,
            IL
        );
        self.elements
            .iter()
            .all(|elem| elem.input[input_ix] != Some(true))
    }

    pub fn is_unate(&self) -> bool {
        (0..IL).all(|input_ix| {
            self.is_monotone_increasing(input_ix) || self.is_monotone_decreasing(input_ix)
        })
    }

    // ---
    // Helper methods
    // ---

    fn union_impl(&self, other: &Self) -> Self {
        let elements = self
            .elements
            .iter()
            .chain(&other.elements)
            .cloned()
            .collect();
        Self { elements }
    }

    fn intersection_impl(&self, other: &Self) -> Self {
        // page 24
        let elements = self
            .elements
            .iter()
            .cartesian_product(&other.elements)
            .filter_map(|(c, d)| (c & d))
            .collect();
        Self { elements }
    }

    fn intersect_and_remove(&mut self, other: &mut Self) -> Self {
        let result: BTreeSet<_> = self
            .elements
            .intersection(&other.elements)
            .cloned()
            .collect();
        self.elements.retain(|p| result.contains(p));
        other.elements.retain(|p| result.contains(p));

        Self { elements: result }
    }
}

impl<const IL: usize, const OL: usize> BitAnd for CubeSet<IL, OL> {
    type Output = CubeSet<IL, OL>;

    fn bitand(self, rhs: Self) -> Self::Output {
        self.intersection_impl(&rhs)
    }
}

impl<'a, const IL: usize, const OL: usize> BitAnd<&'a CubeSet<IL, OL>> for CubeSet<IL, OL> {
    type Output = CubeSet<IL, OL>;

    fn bitand(self, rhs: &'a CubeSet<IL, OL>) -> Self::Output {
        self.intersection_impl(rhs)
    }
}

impl<'a, 'b, const IL: usize, const OL: usize> BitAnd<&'a CubeSet<IL, OL>> for &'b CubeSet<IL, OL> {
    type Output = CubeSet<IL, OL>;

    fn bitand(self, rhs: &'a CubeSet<IL, OL>) -> Self::Output {
        self.intersection_impl(rhs)
    }
}

impl<'b, const IL: usize, const OL: usize> BitAnd<CubeSet<IL, OL>> for &'b CubeSet<IL, OL> {
    type Output = CubeSet<IL, OL>;

    fn bitand(self, rhs: CubeSet<IL, OL>) -> Self::Output {
        self.intersection_impl(&rhs)
    }
}

impl<const IL: usize, const OL: usize> BitOr for CubeSet<IL, OL> {
    type Output = CubeSet<IL, OL>;

    fn bitor(self, rhs: Self) -> Self::Output {
        self.union_impl(&rhs)
    }
}

impl<'a, const IL: usize, const OL: usize> BitOr<&'a CubeSet<IL, OL>> for CubeSet<IL, OL> {
    type Output = CubeSet<IL, OL>;

    fn bitor(self, rhs: &'a CubeSet<IL, OL>) -> Self::Output {
        self.union_impl(rhs)
    }
}

impl<'a, 'b, const IL: usize, const OL: usize> BitOr<&'a CubeSet<IL, OL>> for &'b CubeSet<IL, OL> {
    type Output = CubeSet<IL, OL>;

    fn bitor(self, rhs: &'a CubeSet<IL, OL>) -> Self::Output {
        self.union_impl(rhs)
    }
}

impl<'b, const IL: usize, const OL: usize> BitOr<CubeSet<IL, OL>> for &'b CubeSet<IL, OL> {
    type Output = CubeSet<IL, OL>;

    fn bitor(self, rhs: CubeSet<IL, OL>) -> Self::Output {
        self.union_impl(&rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complement() {
        let cube = Cube::from_numeric([1, 0, 2], [4, 3]).unwrap();
        let complement = Cube::from_numeric([0, 1, 2], [4, 3]).unwrap();
        assert_eq!(!cube, complement);
    }

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

    #[test]
    fn test_cofactor() {
        // examples on page 30
        let cube_set = CubeSet::from_numeric([
            ([1, 1, 0, 2], [4, 4]),
            ([0, 1, 2, 0], [4, 4]),
            ([1, 1, 1, 1], [4, 3]),
        ])
        .unwrap();
        {
            let p = Cube::from_numeric([1, 1, 2, 2], [4, 3]).unwrap();

            let result =
                CubeSet::from_numeric([([2, 2, 0, 2], [4, 4]), ([2, 2, 1, 1], [4, 4])]).unwrap();

            assert_eq!(cube_set.cofactor(&p), result);
        }

        {
            let p = Cube::from_numeric([2, 2, 2, 1], [4, 4]).unwrap();

            let result =
                CubeSet::from_numeric([([1, 1, 0, 2], [4, 4]), ([1, 1, 1, 2], [4, 3])]).unwrap();

            assert_eq!(cube_set.cofactor(&p), result);

            let result_complement =
                CubeSet::from_numeric([([1, 1, 0, 2], [4, 4]), ([0, 1, 2, 2], [4, 4])]).unwrap();

            assert_eq!(cube_set.cofactor(&!(&p)), result_complement);

            let expansion = CubeSet::from_numeric([
                ([1, 1, 0, 1], [4, 4]),
                ([1, 1, 0, 0], [4, 4]),
                ([1, 1, 1, 1], [4, 3]),
                ([0, 1, 2, 0], [4, 4]),
            ])
            .unwrap();
            assert_eq!(cube_set.shannon_expansion(&p), expansion);
        }
    }

    #[test]
    fn test_is_unate() {
        let cube_set = CubeSet::from_numeric([([1, 1, 0], [4]), ([2, 0, 2], [4])]).unwrap();
        assert!(!cube_set.is_unate());

        let cube_set = CubeSet::from_numeric([([1, 2, 0], [4]), ([2, 0, 2], [4])]).unwrap();
        assert!(cube_set.is_unate());
    }
}
