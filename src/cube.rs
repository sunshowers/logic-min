// Copyright (c) The logic-min Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::errors::InvalidCubeNumeric;
use arrayvec::ArrayVec;
use itertools::Itertools;
use std::{
    collections::BTreeSet,
    ops::{BitAnd, BitOr, Not},
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

    pub fn positive_half_space(input_ix: usize) -> Self {
        let mut input = [None; IL];
        input[input_ix] = Some(true);
        let output = [true; OL];

        Self { input, output }
    }

    pub fn negative_half_space(input_ix: usize) -> Self {
        let mut input = [None; IL];
        input[input_ix] = Some(false);
        let output = [true; OL];

        Self { input, output }
    }

    pub fn evaluate(&self, values: &[bool; IL]) -> bool {
        for (variable, value) in self.input.iter().zip(values.iter()) {
            match (variable, value) {
                (Some(v), value) => {
                    if v != value {
                        return false;
                    }
                }
                (None, _) => {}
            }
        }
        true
    }

    pub fn contains(&self, other: &Cube<IL, OL>) -> bool {
        let input_contains = self
            .input
            .iter()
            .zip(&other.input)
            .all(|(&c, &d)| CubeContains::input_contains(c, d) >= CubeContains::Contains);
        let output_contains = self
            .output
            .iter()
            .zip(&other.output)
            .all(|(&c, &d)| CubeContains::output_contains(c, d) >= CubeContains::Contains);
        input_contains && output_contains
    }

    pub fn strictly_contains(&self, other: &Cube<IL, OL>) -> bool {
        let mut any_strictly = false;

        let mut process_contains = |contains: CubeContains| -> bool {
            match contains {
                CubeContains::Strictly => {
                    any_strictly = true;
                    true
                }
                CubeContains::Contains => true,
                CubeContains::DoesNotContain => false,
            }
        };

        let input_contains = self
            .input
            .iter()
            .zip(&other.input)
            .all(|(&c, &d)| process_contains(CubeContains::input_contains(c, d)));

        println!(
            "*** *** *** input {:?} contains {:?}: {}",
            self, other, input_contains
        );

        let output_contains = self
            .output
            .iter()
            .zip(&other.output)
            .all(|(&c, &d)| process_contains(CubeContains::output_contains(c, d)));

        input_contains && output_contains && any_strictly
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

        //println!("*** input: {:?}, output: {:?}", input, output);

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
        println!("** intersecting {:?} with {:?}", self, cube_set);
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

    let any_output_true = if OL == 0 {
        // If there are no outputs, assume this to be true.
        true
    } else {
        let mut res = false;
        for (&c, &d) in a.output.iter().zip(&b.output) {
            let elem = intersect_output(c, d);
            if elem {
                res = true;
            }
            output.push(elem);
        }
        res
    };

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
    let res = match (c, d) {
        (None, d) => IntersectInputResult::Value(d),
        (d, None) => IntersectInputResult::Value(d),
        (Some(false), Some(false)) => IntersectInputResult::Value(Some(false)),
        (Some(true), Some(true)) => IntersectInputResult::Value(Some(true)),
        (Some(false), Some(true)) | (Some(true), Some(false)) => IntersectInputResult::Phi,
    };
    //println!("intersect one result: {:?}", res);
    res
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
            (Some(true), Some(false) | None) => Self::DoesNotContain,
            (Some(true), Some(true)) => Self::Contains,
            (None, Some(true) | Some(false)) => Self::Strictly,
            (None, None) => Self::Contains,
        }
    }

    fn output_contains(c: bool, d: bool) -> Self {
        // page 23
        match (c, d) {
            (false, false) => CubeContains::Contains,
            (false, true) => CubeContains::DoesNotContain,
            (true, false) => CubeContains::Strictly,
            (true, true) => CubeContains::Contains,
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
    pub fn cube_count(&self) -> usize {
        self.elements.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    pub fn evaluate(&self, values: &[bool; IL]) -> bool {
        self.elements.iter().any(|elem| elem.evaluate(values))
    }

    pub fn check_logically_equivalent(&self, other: &Self) -> Result<(), [bool; IL]> {
        // Iterate over all possible (false, true) combinations.
        for input_bits in 0..2_u32.pow(IL as u32) {
            let mut values = [false; IL];
            for bit in 0..IL {
                if (input_bits >> bit) & 1 == 1 {
                    values[bit] = true;
                }
            }
            if self.evaluate(&values) != other.evaluate(&values) {
                return Err(values);
            }
        }
        Ok(())
    }

    pub fn is_cover(&self, c: &Cube<IL, OL>) -> bool {
        unimplemented!("need to implement minterms first")
    }

    #[inline]
    pub fn shannon_expansion(&self, input_ix: usize) -> ShannonExpansion<IL, OL> {
        ShannonExpansion::new(self, input_ix, std::convert::identity)
    }

    #[inline]
    pub fn shannon_expansion_with_transform(
        &self,
        input_ix: usize,
        transform: impl FnMut(CubeSet<IL, OL>) -> CubeSet<IL, OL>,
    ) -> ShannonExpansion<IL, OL> {
        ShannonExpansion::new(self, input_ix, transform)
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

    #[inline]
    pub fn make_unate(self) -> Result<UnateCubeSet<IL, OL>, Self> {
        UnateCubeSet::new(self)
    }

    pub fn make_unate_with_binate_select(self) -> Result<UnateCubeSet<IL, OL>, (Self, usize)> {
        // Number of cubes with Some(true) in the jth input position.
        let mut ones = [0_u32; IL];
        // Number of cubes with Some(false) in the jth input position.
        let mut zeroes = [0_u32; IL];

        for elem in &self.elements {
            for input_ix in 0..IL {
                match elem.input[input_ix] {
                    Some(true) => ones[input_ix] += 1,
                    Some(false) => zeroes[input_ix] += 1,
                    None => {}
                }
            }
        }

        // TODO: combine this loop and the below one.

        let is_unate = zeroes
            .iter()
            .zip(ones.iter())
            .all(|(&zeroes_j, &ones_j)| zeroes_j == 0 || ones_j == 0);
        if is_unate {
            // TODO: compute monotonicity here rather than recomputing it in UnateCubeSet::new
            return Ok(UnateCubeSet::new(self).expect("we already checked unate"));
        }

        let max_binate_ix = zeroes
            .iter()
            .zip(ones.iter())
            .enumerate()
            .filter_map(|(input_ix, (&zeroes_j, &ones_j))| {
                (zeroes_j != 0 && ones_j != 0).then(|| (input_ix, zeroes_j + ones_j))
            })
            .max_by_key(|(_, binate_val)| *binate_val)
            .map(|(max_binate_ix, _)| max_binate_ix)
            .expect("there's at least one input");
        Err((self, max_binate_ix))
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
        self.elements.retain(|p| !result.contains(p));
        other.elements.retain(|p| !result.contains(p));

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

impl<const IL: usize> CubeSet<IL, 0> {
    /// Basic algorithm to simplify a single-output cube set.
    pub fn simplify_basic(self) -> Self {
        println!("starting simplification: {:?}", self);
        match self.make_unate_with_binate_select() {
            Ok(unate_set) => unate_set.simplify().into_inner(),
            Err((cube_set, max_binate_ix)) => {
                println!("max binate ix: {}", max_binate_ix);
                let expansion = cube_set
                    .shannon_expansion_with_transform(max_binate_ix, |cube_set| {
                        cube_set.simplify_basic()
                    });

                println!(
                    "expansion: (input ix: {}), {:?}",
                    expansion.input_ix, expansion
                );
                let merged = expansion.merge_with_containment();
                if merged.cube_count() <= cube_set.cube_count() {
                    merged
                } else {
                    cube_set
                }
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnateCubeSet<const IL: usize, const OL: usize> {
    inner: CubeSet<IL, OL>,
    monotonicity: [Monotonicity; IL],
}

impl<const IL: usize, const OL: usize> UnateCubeSet<IL, OL> {
    /// Checks if `cube_set` is unate, and if so, returns Self.
    ///
    /// Returns `Err(cube_set)` if `cube_set` is not unate.
    pub fn new(cube_set: CubeSet<IL, OL>) -> Result<Self, CubeSet<IL, OL>> {
        let mut monotonicity: ArrayVec<Monotonicity, IL> = ArrayVec::new();
        for input_ix in 0..IL {
            if cube_set.is_monotone_increasing(input_ix) {
                monotonicity.push(Monotonicity::Increasing);
            } else if cube_set.is_monotone_decreasing(input_ix) {
                monotonicity.push(Monotonicity::Decreasing);
            } else {
                return Err(cube_set);
            }
        }

        // SAFETY: we push exactly IL elements
        debug_assert_eq!(monotonicity.len(), monotonicity.capacity());
        let monotonicity = unsafe { monotonicity.into_inner_unchecked() };
        Ok(Self {
            inner: cube_set,
            monotonicity,
        })
    }

    #[inline]
    pub fn as_inner(&self) -> &CubeSet<IL, OL> {
        &self.inner
    }

    #[inline]
    pub fn monotonicity(&self) -> &[Monotonicity; IL] {
        &self.monotonicity
    }

    #[inline]
    pub fn into_inner(self) -> CubeSet<IL, OL> {
        self.inner
    }
}

impl<const IL: usize> UnateCubeSet<IL, 0> {
    /// Simplifies a single-variable unate cube set: removes any elements that are contained in
    /// other elements.
    pub fn simplify(&self) -> Self {
        // TODO: use set ordering more intelligently?
        let simplified: BTreeSet<_> = self
            .inner
            .elements
            .iter()
            .filter(|elem| {
                let contains = self
                    .inner
                    .elements
                    .iter()
                    .any(|contains| contains.strictly_contains(elem));
                !contains
            })
            .cloned()
            .collect();
        // The monotonicity of the simplified set matches that of the original set.
        let monotonicity = self.monotonicity;

        println!("** UNATE SIMPLIFY: {:?}", simplified);

        Self {
            inner: CubeSet {
                elements: simplified,
            },
            monotonicity,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Monotonicity {
    Decreasing,
    Increasing,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShannonExpansion<const IL: usize, const OL: usize> {
    positive: CubeSet<IL, OL>,
    negative: CubeSet<IL, OL>,
    input_ix: usize,
    // Positive and negative half-spaces.
    pos_cube: Cube<IL, OL>,
    neg_cube: Cube<IL, OL>,
}

impl<const IL: usize, const OL: usize> ShannonExpansion<IL, OL> {
    pub fn new(
        cube_set: &CubeSet<IL, OL>,
        input_ix: usize,
        mut transform: impl FnMut(CubeSet<IL, OL>) -> CubeSet<IL, OL>,
    ) -> Self {
        let pos_cube = Cube::positive_half_space(input_ix);
        let neg_cube = Cube::negative_half_space(input_ix);

        let pos_cofactor = cube_set.cofactor(&pos_cube);
        let neg_cofactor = cube_set.cofactor(&neg_cube);

        let transformed_pos = (transform)(pos_cofactor);
        let transformed_neg = (transform)(neg_cofactor);
        Self {
            positive: transformed_pos,
            negative: transformed_neg,
            input_ix,
            pos_cube,
            neg_cube,
        }
    }

    /// Given two subcovers self and other, obtained by the Shannon expansion with
    /// respect to a cube p, computes a cover by merging them.
    pub fn merge_with_identity(mut self) -> CubeSet<IL, OL> {
        let intersection = self.positive.intersect_and_remove(&mut self.negative);
        (&self.pos_cube & &self.positive) | (&self.neg_cube & &self.negative) | intersection
    }

    pub fn merge_with_containment(mut self) -> CubeSet<IL, OL> {
        println!(
            "\n\n*** STARTING MERGE WITH CONTAINMENT: pos: {:?}, neg: {:?}, ix: {}",
            self.positive, self.negative, self.input_ix
        );
        let mut intersection = self.positive.intersect_and_remove(&mut self.negative);

        println!(
            "after direct intersection: pos: {:?}\n  neg: {:?}\n  intersection: {:?}",
            self.positive, self.negative, intersection
        );

        let mut new_pos: CubeSet<IL, OL> = Default::default();
        let mut new_neg: CubeSet<IL, OL> = Default::default();

        for pos_elem in &self.positive.elements {
            for neg_elem in &self.negative.elements {
                if pos_elem.strictly_contains(neg_elem) {
                    println!(
                        "********** strictly contains pos {:?} in neg: {:?}",
                        pos_elem, neg_elem
                    );
                    // For some reason rust-analyzer chokes on neg_elem.clone()
                    intersection.elements.insert(Clone::clone(neg_elem));
                    new_pos.elements.insert(Clone::clone(pos_elem));
                } else if neg_elem.strictly_contains(pos_elem) {
                    intersection.elements.insert(Clone::clone(pos_elem));
                    new_neg.elements.insert(Clone::clone(neg_elem));
                } else {
                    new_pos.elements.insert(Clone::clone(pos_elem));
                    new_neg.elements.insert(Clone::clone(neg_elem));
                }
            }
        }

        println!(
            "### after containment: pos: {:?}\n  neg: {:?}\n  intersection: {:?}\n\n",
            new_pos, new_neg, intersection
        );

        (&self.pos_cube & new_pos) | (&self.neg_cube & new_neg) | intersection
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
            let input_ix = 3;

            let result =
                CubeSet::from_numeric([([1, 1, 0, 2], [4, 4]), ([1, 1, 1, 2], [4, 3])]).unwrap();

            assert_eq!(
                cube_set.cofactor(&Cube::positive_half_space(input_ix)),
                result
            );

            let result_complement =
                CubeSet::from_numeric([([1, 1, 0, 2], [4, 4]), ([0, 1, 2, 2], [4, 4])]).unwrap();

            assert_eq!(
                cube_set.cofactor(&Cube::negative_half_space(input_ix)),
                result_complement
            );

            let expected_positive =
                CubeSet::from_numeric([([1, 1, 0, 1], [4, 4]), ([1, 1, 1, 1], [4, 3])]).unwrap();
            let expected_negative =
                CubeSet::from_numeric([([1, 1, 0, 0], [4, 4]), ([0, 1, 2, 0], [4, 4])]).unwrap();
            let actual = cube_set.shannon_expansion(input_ix);
            assert_eq!(
                actual.positive, expected_positive,
                "positive expansion matches"
            );
            assert_eq!(
                actual.negative, expected_negative,
                "negative expansion matches"
            );
        }
    }

    #[test]
    fn test_is_unate() {
        {
            let cube_set = CubeSet::from_numeric([([1, 1, 0], [4]), ([2, 0, 2], [4])]).unwrap();
            assert!(cube_set.clone().make_unate().is_err());
            assert!(matches!(cube_set.make_unate_with_binate_select(), Err((_, x)) if x == 1));
        }

        {
            let cube_set = CubeSet::from_numeric([([1, 2, 0], [4]), ([2, 0, 2], [4])]).unwrap();
            assert!(cube_set.clone().make_unate().is_ok());
            assert!(cube_set.make_unate_with_binate_select().is_ok());
        }
    }

    #[test]
    fn test_simplify_basic() {
        let cube_set =
            CubeSet::from_numeric([([0, 2, 2], []), ([1, 1, 2], []), ([2, 1, 1], [])]).unwrap();
        let expected = CubeSet::from_numeric([([2, 1, 2], []), ([0, 2, 2], [])]).unwrap();

        // input_ix 0 is the only binate variable.
        let (cube_set, max_binate_ix) = cube_set.make_unate_with_binate_select().unwrap_err();
        assert_eq!(max_binate_ix, 0, "only binate variable");
        let actual = cube_set.simplify_basic();

        assert_eq!(actual, expected, "basic simplification works");
        actual
            .check_logically_equivalent(&expected)
            .expect("check logical equivalence");

        let cube_set = CubeSet::from_numeric([
            ([2, 0, 0, 0, 2, 0], []),
            ([0, 0, 1, 2, 1, 2], []),
            ([1, 2, 1, 2, 2, 1], []),
            ([0, 1, 2, 1, 2, 2], []),
            ([2, 1, 2, 2, 0, 1], []),
            ([2, 1, 0, 0, 2, 1], []),
            ([1, 0, 1, 0, 0, 2], []),
            ([2, 0, 0, 0, 2, 1], []),
            ([1, 0, 1, 0, 1, 2], []),
            ([2, 1, 0, 0, 2, 0], []),
            ([1, 1, 2, 1, 2, 0], []),
            ([1, 1, 2, 1, 2, 1], []),
        ])
        .unwrap();
        let actual = cube_set.simplify_basic();
        println!("ACTUAL: {:?}", actual);
        println!("{}", actual.cube_count());

        let expected = CubeSet::from_numeric([
            ([1, 1, 2, 2, 2, 1], []),
            ([2, 1, 2, 1, 2, 2], []),
            ([2, 1, 2, 2, 0, 1], []),
            ([2, 1, 0, 2, 2, 2], []),
            ([0, 0, 1, 2, 1, 2], []),
            ([1, 0, 2, 0, 2, 2], []),
            ([1, 2, 1, 2, 2, 1], []),
            ([2, 2, 0, 0, 2, 2], []),
        ])
        .unwrap();

        println!("{:?}", actual.check_logically_equivalent(&expected));
    }
}
