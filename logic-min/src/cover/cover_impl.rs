// Copyright (c) The logic-min Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    cover::{CoverAlgebraicDisplay, CoverMatrixDisplay},
    cube::Cube,
    errors::InvalidCubeNumeric,
};
use itertools::Itertools;
use std::{
    collections::BTreeSet,
    ops::{BitAnd, BitOr},
};

use super::caches::{ColumnData, CoverCache};

#[derive(Clone, Debug, Default)]
pub struct Cover<const IL: usize, const OL: usize> {
    elements: CoverElements<IL, OL>,
    cache: CoverCache<IL, OL>,
}

impl<const IL: usize> Cover<IL, 0> {
    pub fn from_numeric0(
        numeric: impl IntoIterator<Item = [u8; IL]>,
    ) -> Result<Self, InvalidCubeNumeric> {
        let elements: BTreeSet<_> = numeric
            .into_iter()
            .map(|input| Cube::from_numeric0(input))
            .collect::<Result<_, _>>()?;
        Ok(Self::new(elements))
    }
}

impl<const IL: usize, const OL: usize> Cover<IL, OL> {
    pub const INPUT_LEN: usize = IL;
    pub const OUTPUT_LEN: usize = OL;

    pub fn new(elements: impl IntoIterator<Item = Cube<IL, OL>>) -> Self {
        Self {
            elements: CoverElements(elements.into_iter().collect()),
            cache: CoverCache::default(),
        }
    }

    pub fn from_numeric(
        numeric: impl IntoIterator<Item = ([u8; IL], [u8; OL])>,
    ) -> Result<Self, InvalidCubeNumeric> {
        let elements: BTreeSet<_> = numeric
            .into_iter()
            .map(|(input, output)| Cube::from_numeric(input, output))
            .collect::<Result<_, _>>()?;
        Ok(Self::new(elements))
    }

    pub fn cofactor(&self, p: &Cube<IL, OL>) -> Self {
        Self::new(self.elements().iter().filter_map(|elem| elem.cofactor(p)))
    }

    #[inline]
    pub fn cube_count(&self) -> usize {
        self.elements().len()
    }

    #[inline]
    pub fn meaningful_input_count(&self) -> usize {
        self.get_or_init_column_cache().0
    }

    #[inline]
    pub fn meaningful_input_ixs(&self) -> impl Iterator<Item = usize> + '_ {
        self.get_column_data()
            .iter()
            .enumerate()
            .filter_map(|(ix, data)| data.is_meaningful().then(|| ix))
    }

    #[inline]
    pub fn elements(&self) -> &BTreeSet<Cube<IL, OL>> {
        &self.elements.0
    }

    #[inline]
    pub fn elements_mut(&mut self) -> &mut BTreeSet<Cube<IL, OL>> {
        self.cache.invalidate();
        &mut self.elements.0
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.elements().is_empty()
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
        self.elements()
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

    /// Returns the result of evaluating all the output values against the input value, specified in
    /// terms of meaningful column indexes.
    ///
    /// Panics if:
    /// * the length of the slice is different from `self.meaningful_column_count()`
    /// * `OL == 0`. Use `evaluate0_meaningful` to get a true/false answer when OL == 0.
    pub fn evaluate_meaningful(&self, values: &[bool]) -> [bool; OL] {
        assert_ne!(
            OL, 0,
            "output length {} must not be 0 -- use evaluate0_meaningful to get an answer when it is 0",
            OL
        );
        let meaningful_ixs: Vec<_> = self.meaningful_input_ixs().collect();
        let mut res = [false; OL];
        for output_ix in 0..OL {
            res[output_ix] = self
                .output_component(output_ix)
                .any(|elem| elem.evaluate0_ixs(values, &meaningful_ixs));
        }
        res
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
        self.get_column_data()[input_ix].is_monotone_increasing()
    }

    pub fn is_monotone_decreasing(&self, input_ix: usize) -> bool {
        assert!(
            input_ix < IL,
            "input elem {} must be in range [0..{})",
            input_ix,
            IL
        );
        self.get_column_data()[input_ix].is_monotone_decreasing()
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

        for elem in self.elements() {
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
        let simplified = self
            .elements()
            .iter()
            .filter(|elem| {
                let contains = self
                    .elements()
                    .iter()
                    .any(|contains| contains.strictly_contains(elem));
                !contains
            })
            .cloned();
        Self::new(simplified)
    }

    pub fn consensus(&self, other: &Self) -> Self {
        let elements = self
            .elements()
            .iter()
            .cartesian_product(other.elements())
            .filter_map(|(c, d)| {
                let res = c.consensus(d);
                res
            });
        Self::new(elements)
    }

    // ---
    // Helper methods
    // ---

    fn union_impl(&self, other: &Self) -> Self {
        let elements = self.elements().iter().chain(other.elements()).cloned();
        Self::new(elements)
    }

    fn intersection_impl(&self, other: &Self) -> Self {
        // page 24
        let elements = self
            .elements()
            .iter()
            .cartesian_product(other.elements())
            .filter_map(|(c, d)| (c & d));
        Self::new(elements)
    }

    fn intersect_and_remove(&mut self, other: &mut Self) -> Self {
        let result: BTreeSet<_> = self
            .elements()
            .intersection(other.elements())
            .cloned()
            .collect();
        self.elements_mut().retain(|p| !result.contains(p));
        other.elements_mut().retain(|p| !result.contains(p));
        Self::new(result)
    }

    #[inline]
    fn get_column_data(&self) -> &[ColumnData; IL] {
        self.get_or_init_column_cache().1
    }

    #[inline]
    fn get_or_init_column_cache(&self) -> (usize, &[ColumnData; IL]) {
        self.cache.get_or_init_column_data(self.elements())
    }
}

impl<const IL: usize> Cover<IL, 0> {
    pub fn evaluate0(&self, values: &[bool; IL]) -> bool {
        self.elements().iter().any(|elem| elem.evaluate0(values))
    }

    pub fn evaluate0_meaningful(&self, values: &[bool]) -> bool {
        let meaningful_ixs: Vec<_> = self.meaningful_input_ixs().collect();
        self.elements()
            .iter()
            .any(|elem| elem.evaluate0_ixs(values, &meaningful_ixs))
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

impl<const IL: usize, const OL: usize> PartialEq for Cover<IL, OL> {
    fn eq(&self, other: &Self) -> bool {
        &self.elements == &other.elements
    }
}

impl<const IL: usize, const OL: usize> Eq for Cover<IL, OL> {}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct CoverElements<const IL: usize, const OL: usize>(BTreeSet<Cube<IL, OL>>);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnateCover<const IL: usize, const OL: usize> {
    inner: Cover<IL, OL>,
}

impl<const IL: usize, const OL: usize> UnateCover<IL, OL> {
    /// Checks if `cover` is unate, and if so, returns Self.
    ///
    /// Returns `Err(cover)` if `cover` is not unate.
    pub fn new(cover: Cover<IL, OL>) -> Result<Self, Cover<IL, OL>> {
        let column_data = cover.get_column_data();
        if !column_data.iter().all(|data| data.is_unate()) {
            return Err(cover);
        }

        Ok(Self { inner: cover })
    }

    /// Simplifies a unate cover by removing any elements that are contained in other elements.
    pub fn simplify(&self) -> Self {
        let inner = self.inner.single_cube_containment();

        // The simplification of a unate cover is also unate.
        Self { inner }
    }

    #[inline]
    pub fn as_inner(&self) -> &Cover<IL, OL> {
        &self.inner
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
        let mut intersection = self.positive.intersect_and_remove(&mut self.negative);
        let intersection_elements = intersection.elements_mut();

        let mut new_pos: Cover<IL, OL> = self.positive.clone();
        let new_pos_elements = new_pos.elements_mut();
        let mut new_neg: Cover<IL, OL> = self.negative.clone();
        let new_neg_elements = new_neg.elements_mut();

        for pos_elem in self.positive.elements() {
            for neg_elem in self.negative.elements() {
                if pos_elem.strictly_contains(neg_elem) {
                    // For some reason rust-analyzer chokes on neg_elem.clone()
                    intersection_elements.insert(Clone::clone(neg_elem));
                    new_neg_elements.remove(neg_elem);
                } else if neg_elem.strictly_contains(pos_elem) {
                    intersection_elements.insert(Clone::clone(pos_elem));
                    new_pos_elements.remove(pos_elem);
                }
            }
        }

        (&self.pos_half_space & new_pos) | (&self.neg_half_space & new_neg) | intersection
    }
}

// ---
// Displayers
// ---

#[cfg(test)]
mod tests {
    use super::*;

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
