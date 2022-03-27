// Copyright (c) The logic-min Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    cube::{AlgebraicSymbol, Cube, MatrixDisplayFormat},
    errors::InvalidCubeNumeric,
};
use arrayvec::ArrayVec;
use itertools::{Itertools, Position};
use std::{
    borrow::Cow,
    cmp::Ordering,
    collections::BTreeSet,
    fmt,
    ops::{BitAnd, BitOr},
};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Cover<const IL: usize, const OL: usize> {
    pub elements: BTreeSet<Cube<IL, OL>>,
}

impl<const IL: usize> Cover<IL, 0> {
    pub fn from_numeric0(
        numeric: impl IntoIterator<Item = [u8; IL]>,
    ) -> Result<Self, InvalidCubeNumeric> {
        let elements = numeric
            .into_iter()
            .map(|input| Cube::from_numeric0(input))
            .collect::<Result<_, _>>()?;
        Ok(Self { elements })
    }
}

impl<const IL: usize, const OL: usize> Cover<IL, OL> {
    pub const INPUT_LEN: usize = IL;
    pub const OUTPUT_LEN: usize = OL;

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

    #[inline]
    pub fn matrix_display(&self) -> CoverMatrixDisplay<'_, IL, OL> {
        CoverMatrixDisplay::new(self)
    }

    #[inline]
    pub fn algebraic_display(&self) -> CoverAlgebraicDisplay<'_, IL, OL> {
        CoverAlgebraicDisplay::new(self)
    }

    #[inline]
    pub fn try_into_cubeset0(self) -> Result<Cover<IL, 0>, Cover<IL, OL>> {
        if OL == 0 {
            // SAFETY: `BTreeSet<IL, 0>` has exactly the same layout as `BTreeSet<IL, OL>` when OL
            // == 0
            Ok(unsafe { std::mem::transmute::<Cover<IL, OL>, Cover<IL, 0>>(self) })
        } else {
            Err(self)
        }
    }

    #[inline]
    pub fn try_as_cover0(&self) -> Option<&Cover<IL, 0>> {
        if OL == 0 {
            // SAFETY: `BTreeSet<IL, 0>` has exactly the same layout as `BTreeSet<IL, OL>` when OL
            // == 0
            Some(unsafe { &*(self as *const Cover<IL, OL> as *const Cover<IL, 0>) })
        } else {
            None
        }
    }

    /// Returns the part of this [`Cover`] which resolves to true for the given input.
    pub fn output_component(&self, output_ix: usize) -> impl Iterator<Item = &Cube<IL, 0>> + '_ {
        assert!(
            output_ix < OL,
            "output ix {} must be in range 0..{}",
            output_ix,
            OL
        );
        self.elements
            .iter()
            .filter_map(move |elem| elem.output[output_ix].then(|| elem.as_input_cube()))
    }

    /// Returns the result of evaluating all the output values against the input value.
    ///
    /// Panics if `OL == 0`. Use `evaluate0` to get a true/false answer when OL == 0.
    pub fn evaluate(&self, values: &[bool; IL]) -> [bool; OL] {
        assert_ne!(
            OL, 0,
            "output length {} must not be 0 -- use evaluate0 to get an answer when it is 0",
            OL
        );
        let mut res = [false; OL];
        for output_ix in 0..OL {
            res[output_ix] = self
                .output_component(output_ix)
                .any(|elem| elem.evaluate0(values));
        }

        res
    }

    pub fn is_tautology(&self) -> bool {
        // TODO: unate reduction/component reduction
        match self.try_as_cover0() {
            Some(cover0) => cover0.is_tautology0(),
            None => {
                // Check that each output component is tautological.
                for output_ix in 0..OL {
                    let component = Cover::new(self.output_component(output_ix).cloned());
                    if !component.is_tautology0() {
                        return false;
                    }
                }
                true
            }
        }
    }

    pub fn check_logically_equivalent(&self, other: &Self) -> Result<(), [bool; IL]> {
        match (self.try_as_cover0(), other.try_as_cover0()) {
            (Some(self0), Some(other0)) => self0.check_logically_equivalent0(other0),
            (None, None) => {
                // TODO: benchmark against breaking up separately
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
            _ => unreachable!("self and other have the same OL so this isn't possible"),
        }
    }

    #[inline]
    pub fn shannon_expansion(&self, split_ix: usize) -> ShannonExpansion<IL, OL> {
        ShannonExpansion::new(self, split_ix)
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
    pub fn make_unate(self) -> Result<UnateCover<IL, OL>, Self> {
        UnateCover::new(self)
    }

    pub fn make_unate_or_select_binate(self) -> Result<UnateCover<IL, OL>, (Self, usize)> {
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
            // TODO: compute monotonicity here rather than recomputing it in UnateCover::new
            return Ok(UnateCover::new(self).expect("we already checked unate"));
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

    pub fn single_cube_containment(&self) -> Self {
        let simplified: BTreeSet<_> = self
            .elements
            .iter()
            .filter(|elem| {
                let contains = self
                    .elements
                    .iter()
                    .any(|contains| contains.strictly_contains(elem));
                !contains
            })
            .cloned()
            .collect();
        Self {
            elements: simplified,
        }
    }

    pub fn consensus(&self, other: &Self) -> Self {
        let elements = self
            .elements
            .iter()
            .cartesian_product(&other.elements)
            .filter_map(|(c, d)| {
                let res = c.consensus(d);
                res
            })
            .collect();
        Self { elements }
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

impl<const IL: usize> Cover<IL, 0> {
    pub fn evaluate0(&self, values: &[bool; IL]) -> bool {
        self.elements.iter().any(|elem| elem.evaluate0(values))
    }

    pub fn check_logically_equivalent0(&self, other: &Self) -> Result<(), [bool; IL]> {
        // Iterate over all possible (false, true) combinations.
        for input_bits in 0..2_u32.pow(IL as u32) {
            let mut values = [false; IL];
            for bit in 0..IL {
                if (input_bits >> bit) & 1 == 1 {
                    values[bit] = true;
                }
            }
            if self.evaluate0(&values) != other.evaluate0(&values) {
                return Err(values);
            }
        }
        Ok(())
    }

    pub fn is_tautology0(&self) -> bool {
        // The empty cube is not a tautology.
        if self.elements.is_empty() {
            return false;
        }

        // The tautological cube is present in this cover.
        if self.elements.contains(&Cube::total_universe()) {
            return true;
        }

        // There's a column of all 1s or all 0s.
        'outer: for input_ix in 0..IL {
            let mut any_0s = false;
            let mut any_1s = false;
            for elem in &self.elements {
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

impl<const IL: usize, const OL: usize> BitAnd for Cover<IL, OL> {
    type Output = Cover<IL, OL>;

    fn bitand(self, rhs: Self) -> Self::Output {
        self.intersection_impl(&rhs)
    }
}

impl<'a, const IL: usize, const OL: usize> BitAnd<&'a Cover<IL, OL>> for Cover<IL, OL> {
    type Output = Cover<IL, OL>;

    fn bitand(self, rhs: &'a Cover<IL, OL>) -> Self::Output {
        self.intersection_impl(rhs)
    }
}

impl<'a, 'b, const IL: usize, const OL: usize> BitAnd<&'a Cover<IL, OL>> for &'b Cover<IL, OL> {
    type Output = Cover<IL, OL>;

    fn bitand(self, rhs: &'a Cover<IL, OL>) -> Self::Output {
        self.intersection_impl(rhs)
    }
}

impl<'b, const IL: usize, const OL: usize> BitAnd<Cover<IL, OL>> for &'b Cover<IL, OL> {
    type Output = Cover<IL, OL>;

    fn bitand(self, rhs: Cover<IL, OL>) -> Self::Output {
        self.intersection_impl(&rhs)
    }
}

impl<const IL: usize, const OL: usize> BitOr for Cover<IL, OL> {
    type Output = Cover<IL, OL>;

    fn bitor(self, rhs: Self) -> Self::Output {
        self.union_impl(&rhs)
    }
}

impl<'a, const IL: usize, const OL: usize> BitOr<&'a Cover<IL, OL>> for Cover<IL, OL> {
    type Output = Cover<IL, OL>;

    fn bitor(self, rhs: &'a Cover<IL, OL>) -> Self::Output {
        self.union_impl(rhs)
    }
}

impl<'a, 'b, const IL: usize, const OL: usize> BitOr<&'a Cover<IL, OL>> for &'b Cover<IL, OL> {
    type Output = Cover<IL, OL>;

    fn bitor(self, rhs: &'a Cover<IL, OL>) -> Self::Output {
        self.union_impl(rhs)
    }
}

impl<'b, const IL: usize, const OL: usize> BitOr<Cover<IL, OL>> for &'b Cover<IL, OL> {
    type Output = Cover<IL, OL>;

    fn bitor(self, rhs: Cover<IL, OL>) -> Self::Output {
        self.union_impl(&rhs)
    }
}

impl<const IL: usize> Cover<IL, 0> {
    /// Basic algorithm to simplify a single-output cover.
    pub fn simplify_basic(self) -> Self {
        match self.make_unate_or_select_binate() {
            Ok(unate_set) => unate_set.simplify().into_inner(),
            Err((cover, max_binate_ix)) => {
                println!("max binate ix: {}", max_binate_ix);
                let mut expansion = cover.shannon_expansion(max_binate_ix);
                expansion.transform(|cover| cover.simplify_basic());

                println!(
                    "expansion: (input ix: {}), {:?}",
                    expansion.input_ix, expansion
                );
                let merged = expansion.merge_with_containment();
                if merged.cube_count() <= cover.cube_count() {
                    merged
                } else {
                    cover
                }
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnateCover<const IL: usize, const OL: usize> {
    inner: Cover<IL, OL>,
    monotonicity: [Monotonicity; IL],
}

impl<const IL: usize, const OL: usize> UnateCover<IL, OL> {
    /// Checks if `cover` is unate, and if so, returns Self.
    ///
    /// Returns `Err(cover)` if `cover` is not unate.
    pub fn new(cover: Cover<IL, OL>) -> Result<Self, Cover<IL, OL>> {
        let mut monotonicity: ArrayVec<Monotonicity, IL> = ArrayVec::new();
        for input_ix in 0..IL {
            if cover.is_monotone_increasing(input_ix) {
                monotonicity.push(Monotonicity::Increasing);
            } else if cover.is_monotone_decreasing(input_ix) {
                monotonicity.push(Monotonicity::Decreasing);
            } else {
                return Err(cover);
            }
        }

        // SAFETY: we push exactly IL elements
        debug_assert_eq!(monotonicity.len(), monotonicity.capacity());
        let monotonicity = unsafe { monotonicity.into_inner_unchecked() };
        Ok(Self {
            inner: cover,
            monotonicity,
        })
    }

    /// Simplifies a unate cover by removing any elements that are contained in other elements.
    pub fn simplify(&self) -> Self {
        let inner = self.inner.single_cube_containment();
        // The monotonicity of the simplified set matches that of the original set.
        let monotonicity = self.monotonicity;

        Self {
            inner,
            monotonicity,
        }
    }

    #[inline]
    pub fn as_inner(&self) -> &Cover<IL, OL> {
        &self.inner
    }

    #[inline]
    pub fn monotonicity(&self) -> &[Monotonicity; IL] {
        &self.monotonicity
    }

    #[inline]
    pub fn into_inner(self) -> Cover<IL, OL> {
        self.inner
    }

    #[inline]
    pub fn matrix_display(&self) -> CoverMatrixDisplay<'_, IL, OL> {
        self.inner.matrix_display()
    }

    #[inline]
    pub fn algebraic_display(&self) -> CoverAlgebraicDisplay<'_, IL, OL> {
        self.inner.algebraic_display()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Monotonicity {
    Decreasing,
    Increasing,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShannonExpansion<const IL: usize, const OL: usize> {
    positive: Cover<IL, OL>,
    negative: Cover<IL, OL>,
    input_ix: usize,
    // Positive and negative half-spaces.
    pos_half_space: Cube<IL, OL>,
    neg_half_space: Cube<IL, OL>,
}

impl<const IL: usize, const OL: usize> ShannonExpansion<IL, OL> {
    pub fn new(cover: &Cover<IL, OL>, split_ix: usize) -> Self {
        let pos_half_space = Cube::positive_half_space(split_ix);
        let neg_half_space = Cube::negative_half_space(split_ix);

        let positive = cover.cofactor(&pos_half_space);
        let negative = cover.cofactor(&neg_half_space);
        Self {
            positive,
            negative,
            input_ix: split_ix,
            pos_half_space,
            neg_half_space,
        }
    }

    #[inline]
    pub fn positive(&self) -> &Cover<IL, OL> {
        &self.positive
    }

    #[inline]
    pub fn negative(&self) -> &Cover<IL, OL> {
        &self.negative
    }

    pub fn transform(&mut self, mut f: impl FnMut(Cover<IL, OL>) -> Cover<IL, OL>) {
        let old_positive = std::mem::replace(&mut self.positive, Cover::<IL, OL>::default());
        self.positive = (f)(old_positive);

        let old_negative = std::mem::replace(&mut self.negative, Cover::<IL, OL>::default());
        self.negative = (f)(old_negative);
    }

    #[inline]
    pub fn pos_half_space(&self) -> &Cube<IL, OL> {
        &self.pos_half_space
    }

    #[inline]
    pub fn neg_half_space(&self) -> &Cube<IL, OL> {
        &self.neg_half_space
    }

    /// Given two subcovers self and other, obtained by the Shannon expansion with
    /// respect to a cube p, computes a cover by merging them.
    pub fn merge_with_identity(mut self) -> Cover<IL, OL> {
        let intersection = self.positive.intersect_and_remove(&mut self.negative);
        (&self.pos_half_space & &self.positive)
            | (&self.neg_half_space & &self.negative)
            | intersection
    }

    pub fn merge_with_containment(mut self) -> Cover<IL, OL> {
        println!(
            "\n\n*** STARTING MERGE WITH CONTAINMENT: pos: {:?}, neg: {:?}, ix: {}",
            self.positive, self.negative, self.input_ix
        );
        let mut intersection = self.positive.intersect_and_remove(&mut self.negative);

        println!(
            "after direct intersection: pos: {:?}\n  neg: {:?}\n  intersection: {:?}",
            self.positive, self.negative, intersection
        );

        let mut new_pos: Cover<IL, OL> = self.positive.clone();
        let mut new_neg: Cover<IL, OL> = self.negative.clone();

        for pos_elem in &self.positive.elements {
            for neg_elem in &self.negative.elements {
                if pos_elem.strictly_contains(neg_elem) {
                    // For some reason rust-analyzer chokes on neg_elem.clone()
                    intersection.elements.insert(Clone::clone(neg_elem));
                    new_neg.elements.remove(neg_elem);
                } else if neg_elem.strictly_contains(pos_elem) {
                    intersection.elements.insert(Clone::clone(pos_elem));
                    new_pos.elements.remove(pos_elem);
                }
            }
        }

        println!(
            "### after containment: pos: {:?}\n  neg: {:?}\n  intersection: {:?}\n\n",
            new_pos, new_neg, intersection
        );

        (&self.pos_half_space & new_pos) | (&self.neg_half_space & new_neg) | intersection
    }
}

// ---
// Displayers
// ---

#[derive(Clone, Debug)]
pub struct CoverMatrixDisplay<'a, const IL: usize, const OL: usize> {
    cover: &'a Cover<IL, OL>,
    format: MatrixDisplayFormat,
    internal_separator: Cow<'a, str>,
    input_output_separator: Cow<'a, str>,
    cube_separator: (Cow<'a, str>, bool),
}

impl<'a, const IL: usize, const OL: usize> CoverMatrixDisplay<'a, IL, OL> {
    pub fn new(cover: &'a Cover<IL, OL>) -> Self {
        Self {
            cover,
            format: MatrixDisplayFormat::default(),
            internal_separator: Cow::Borrowed(" "),
            input_output_separator: Cow::Borrowed(" | "),
            cube_separator: (Cow::Borrowed("\n"), true),
        }
    }

    pub fn with_format(mut self, format: MatrixDisplayFormat) -> Self {
        self.format = format;
        self
    }

    pub fn with_internal_separator(mut self, separator: impl Into<Cow<'a, str>>) -> Self {
        self.internal_separator = separator.into();
        self
    }

    pub fn with_input_output_separator(mut self, separator: impl Into<Cow<'a, str>>) -> Self {
        self.input_output_separator = separator.into();
        self
    }

    pub fn with_cube_separator(
        mut self,
        separator: impl Into<Cow<'a, str>>,
        print_last: bool,
    ) -> Self {
        self.cube_separator = (separator.into(), print_last);
        self
    }
}

impl<'a, const IL: usize, const OL: usize> fmt::Display for CoverMatrixDisplay<'a, IL, OL> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let cube_count = self.cover.cube_count();
        for (elem_ix, elem) in self.cover.elements.iter().enumerate() {
            let cube_display = elem
                .matrix_display()
                .with_format(self.format)
                .with_internal_separator(&*self.internal_separator)
                .with_input_output_separator(&*self.input_output_separator);
            write!(f, "{}", cube_display)?;

            let (cube_separator, print_last) = &self.cube_separator;
            if *print_last || elem_ix < cube_count - 1 {
                write!(f, "{}", cube_separator)?;
            }
        }

        Ok(())
    }
}

pub struct CoverAlgebraicDisplay<'a, const IL: usize, const OL: usize> {
    cover: &'a Cover<IL, OL>,
    separator: (Cow<'a, str>, bool),
}

impl<'a, const IL: usize, const OL: usize> CoverAlgebraicDisplay<'a, IL, OL> {
    pub fn new(cover: &'a Cover<IL, OL>) -> Self {
        Self {
            cover,
            separator: (Cow::Borrowed("\n"), true),
        }
    }

    pub fn with_separator(mut self, separator: impl Into<Cow<'a, str>>, print_last: bool) -> Self {
        self.separator = (separator.into(), print_last);
        self
    }
}

impl<'a, const IL: usize, const OL: usize> fmt::Display for CoverAlgebraicDisplay<'a, IL, OL> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.cover.try_as_cover0() {
            Some(cover0) => algebraic_display0(cover0.elements.iter().collect(), f),
            None => {
                let (separator, print_last) = &self.separator;
                // For each output value, print out the corresponding cubes in the component.
                for output_ix in 0..OL {
                    write!(f, "{} = ", AlgebraicSymbol::output(output_ix))?;
                    algebraic_display0(self.cover.output_component(output_ix).collect(), f)?;
                    if output_ix < OL - 1 || *print_last {
                        write!(f, "{}", separator)?;
                    }
                }
                Ok(())
            }
        }
    }
}

fn algebraic_display0<const IL: usize>(
    mut elements: Vec<&Cube<IL, 0>>,
    f: &mut fmt::Formatter,
) -> fmt::Result {
    // Sort the elements lexicographically in the order [Some(true), Some(false), None],
    // This results in minterms starting with `a` showing up first, then `a'`, then
    // minterms not containing a.
    elements.sort_unstable_by(|a, b| {
        for input_ix in 0..IL {
            match (a.input[input_ix], b.input[input_ix]) {
                (Some(true), Some(true)) | (Some(false), Some(false)) | (None, None) => continue,
                (Some(true), Some(false) | None) => return Ordering::Less,
                (Some(false) | None, Some(true)) => return Ordering::Greater,
                (Some(false), None) => return Ordering::Less,
                (None, Some(false)) => return Ordering::Greater,
            }
        }
        Ordering::Equal
    });

    for elem in elements.into_iter().with_position() {
        match elem {
            Position::First(cube) | Position::Middle(cube) => {
                write!(f, "{} + ", cube.algebraic_display())?;
            }
            Position::Last(cube) | Position::Only(cube) => {
                write!(f, "{}", cube.algebraic_display())?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tautology() {
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

    #[test]
    fn test_cofactor() {
        // examples on page 30
        let cover = Cover::from_numeric([
            ([1, 1, 0, 2], [4, 4]),
            ([0, 1, 2, 0], [4, 4]),
            ([1, 1, 1, 1], [4, 3]),
        ])
        .unwrap();
        {
            let p = Cube::from_numeric([1, 1, 2, 2], [4, 3]).unwrap();

            let result =
                Cover::from_numeric([([2, 2, 0, 2], [4, 4]), ([2, 2, 1, 1], [4, 4])]).unwrap();

            assert_eq!(cover.cofactor(&p), result);
        }

        {
            let input_ix = 3;

            let result =
                Cover::from_numeric([([1, 1, 0, 2], [4, 4]), ([1, 1, 1, 2], [4, 3])]).unwrap();

            assert_eq!(cover.cofactor(&Cube::positive_half_space(input_ix)), result);

            let result_complement =
                Cover::from_numeric([([1, 1, 0, 2], [4, 4]), ([0, 1, 2, 2], [4, 4])]).unwrap();

            assert_eq!(
                cover.cofactor(&Cube::negative_half_space(input_ix)),
                result_complement
            );

            let expected_positive =
                Cover::from_numeric([([1, 1, 0, 1], [4, 4]), ([1, 1, 1, 1], [4, 3])]).unwrap();
            let expected_negative =
                Cover::from_numeric([([1, 1, 0, 0], [4, 4]), ([0, 1, 2, 0], [4, 4])]).unwrap();
            let actual = cover.shannon_expansion(input_ix);
            assert_eq!(
                actual.pos_half_space() & actual.positive(),
                expected_positive,
                "positive expansion matches"
            );
            assert_eq!(
                actual.neg_half_space() & actual.negative(),
                expected_negative,
                "negative expansion matches"
            );
        }
    }

    #[test]
    fn test_is_unate() {
        {
            let cover = Cover::from_numeric([([1, 1, 0], [4]), ([2, 0, 2], [4])]).unwrap();
            assert!(cover.clone().make_unate().is_err());
            assert!(matches!(cover.make_unate_or_select_binate(), Err((_, x)) if x == 1));
        }

        {
            let cover = Cover::from_numeric([([1, 2, 0], [4]), ([2, 0, 2], [4])]).unwrap();
            assert!(cover.clone().make_unate().is_ok());
            assert!(cover.make_unate_or_select_binate().is_ok());
        }
    }

    #[test]
    fn test_simplify_basic() {
        let cover =
            Cover::from_numeric([([0, 2, 2], []), ([1, 1, 2], []), ([2, 1, 1], [])]).unwrap();
        let expected = Cover::from_numeric([([2, 1, 2], []), ([0, 2, 2], [])]).unwrap();

        // input_ix 0 is the only binate variable.
        let (cover, max_binate_ix) = cover.make_unate_or_select_binate().unwrap_err();
        assert_eq!(max_binate_ix, 0, "only binate variable");
        let actual = cover.simplify_basic();

        assert_eq!(actual, expected, "basic simplification works");
        actual
            .check_logically_equivalent(&expected)
            .expect("check logical equivalence for simple problem");

        let cover = Cover::from_numeric([
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

        let expected_positive_shannon = Cover::from_numeric([
            ([1, 2, 1, 2, 2, 1], []),
            ([0, 2, 2, 1, 2, 2], []),
            ([2, 2, 2, 2, 0, 1], []),
            ([2, 2, 0, 0, 2, 1], []),
            ([2, 2, 0, 0, 2, 0], []),
            ([1, 2, 2, 1, 2, 0], []),
            ([1, 2, 2, 1, 2, 1], []),
        ])
        .unwrap();
        let shannon = cover.shannon_expansion(1);
        assert_eq!(
            shannon.positive(),
            &expected_positive_shannon,
            "positive shannon matches"
        );

        let simplified_positive_shannon = Cover::from_numeric([
            ([1, 2, 2, 2, 2, 1], []),
            ([2, 2, 2, 1, 2, 2], []),
            ([2, 2, 2, 2, 0, 1], []),
            ([2, 2, 0, 2, 2, 2], []),
        ])
        .unwrap();
        let actual = shannon.positive.simplify_basic();
        assert_eq!(
            actual, simplified_positive_shannon,
            "simplified positive shannon matches"
        );
        simplified_positive_shannon
            .check_logically_equivalent(&actual)
            .expect("check logical equivalence for medium problem");

        // Full problem
        let actual = cover.simplify_basic();
        let expected = Cover::from_numeric([
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
        assert_eq!(actual, expected, "simplified matches for full problem");
        actual
            .check_logically_equivalent(&expected)
            .expect("check logical equivalence for full problem");
    }
}
