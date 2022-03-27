// Copyright (c) The logic-min Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{cover::Cover, errors::InvalidCubeNumeric};
use arrayvec::ArrayVec;
use std::{
    borrow::Cow,
    fmt,
    ops::{BitAnd, Not},
};

#[derive(Clone, Debug, PartialEq, Eq, Ord, PartialOrd)]
pub struct Cube<const IL: usize, const OL: usize> {
    pub input: [Option<bool>; IL],
    pub output: [bool; OL],
}

impl<const IL: usize> Cube<IL, 0> {
    #[inline]
    pub fn new0(input: [Option<bool>; IL]) -> Self {
        Self { input, output: [] }
    }

    #[inline]
    pub fn from_numeric0(input_numeric: [u8; IL]) -> Result<Self, InvalidCubeNumeric> {
        Self::from_numeric(input_numeric, [])
    }
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

    #[inline]
    pub fn matrix_display(&self) -> CubeMatrixDisplay<'_, IL, OL> {
        CubeMatrixDisplay::new(self)
    }

    #[inline]
    pub fn algebraic_display(&self) -> CubeAlgebraicDisplay<'_, IL, OL> {
        CubeAlgebraicDisplay::new(self)
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

        // println!(
        //     "*** *** *** input {:?} contains {:?}: {}",
        //     self, other, input_contains
        // );

        let output_contains = self
            .output
            .iter()
            .zip(&other.output)
            .all(|(&c, &d)| process_contains(CubeContains::output_contains(c, d)));

        input_contains && output_contains && any_strictly
    }

    pub fn is_minterm(&self, output_ix: usize) -> bool {
        // TODO: check that output_ix is between 0 and usize
        let input_minterm = self.input.iter().all(|&c| c.is_some());
        let output_minterm = {
            self.output
                .iter()
                .enumerate()
                .all(|(idx, &c)| if idx == output_ix { c } else { !c })
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

    /// Returns the output distance between `self` and `other`.
    ///
    /// The output distance is defined as 1 if for every output variable at index `ix`,
    /// `self.output[ix] && other.output[ix]` is false.
    ///
    /// # Examples
    ///
    /// ```
    /// use logic_min::cube::Cube;
    ///
    /// let cube1 = Cube::from_numeric([0, 1, 2], [3, 4]).unwrap();
    /// let cube2 = Cube::from_numeric([0, 1, 2], [4, 3]).unwrap();
    /// let cube3 = Cube::from_numeric([0, 1, 2], [4, 4]).unwrap();
    ///
    /// assert_eq!(cube1.output_distance(&cube2), 1);
    /// assert_eq!(cube1.output_distance(&cube3), 0);
    /// ```
    ///
    /// If other is 0, `output_distance` is 0.
    ///
    /// ```
    /// use logic_min::cube::Cube;
    ///
    /// let cube1 = Cube::from_numeric0([0, 1, 2]).unwrap();
    /// let cube2 = Cube::from_numeric0([2, 1, 0]).unwrap();
    ///
    /// assert_eq!(cube1.output_distance(&cube2), 0);
    /// ```
    pub fn output_distance(&self, other: &Cube<IL, OL>) -> usize {
        // page 25
        if OL == 0 {
            // Special case -- no outputs means the output distance is 0.
            0
        } else if self
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

    pub fn is_implicant_of(&self, cover: &Cover<IL, OL>) -> bool {
        (self & cover).is_empty()
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

impl<const IL: usize> Cube<IL, 0> {
    pub fn evaluate0(&self, values: &[bool; IL]) -> bool {
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

impl<'a, const IL: usize, const OL: usize> BitAnd<&'a Cover<IL, OL>> for Cube<IL, OL> {
    type Output = Cover<IL, OL>;

    fn bitand(self, rhs: &'a Cover<IL, OL>) -> Self::Output {
        self.intersect_cover_impl(rhs)
    }
}

impl<'a, 'b, const IL: usize, const OL: usize> BitAnd<&'a Cover<IL, OL>> for &'b Cube<IL, OL> {
    type Output = Cover<IL, OL>;

    fn bitand(self, rhs: &'a Cover<IL, OL>) -> Self::Output {
        self.intersect_cover_impl(rhs)
    }
}

impl<const IL: usize, const OL: usize> BitAnd<Cover<IL, OL>> for Cube<IL, OL> {
    type Output = Cover<IL, OL>;

    fn bitand(self, rhs: Cover<IL, OL>) -> Self::Output {
        self.intersect_cover_impl(&rhs)
    }
}

impl<'b, const IL: usize, const OL: usize> BitAnd<Cover<IL, OL>> for &'b Cube<IL, OL> {
    type Output = Cover<IL, OL>;

    fn bitand(self, rhs: Cover<IL, OL>) -> Self::Output {
        self.intersect_cover_impl(&rhs)
    }
}

impl<const IL: usize, const OL: usize> Cube<IL, OL> {
    fn intersect_cover_impl(&self, cover: &Cover<IL, OL>) -> Cover<IL, OL> {
        Cover::new(cover.elements.iter().filter_map(|c| self & c))
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

pub struct CubeMatrixDisplay<'a, const IL: usize, const OL: usize> {
    cube: &'a Cube<IL, OL>,
    format: MatrixDisplayFormat,
    internal_separator: Cow<'a, str>,
    input_output_separator: Cow<'a, str>,
}

impl<'a, const IL: usize, const OL: usize> CubeMatrixDisplay<'a, IL, OL> {
    pub fn new(cube: &'a Cube<IL, OL>) -> Self {
        Self {
            cube,
            format: MatrixDisplayFormat::default(),
            internal_separator: Cow::Borrowed(" "),
            input_output_separator: Cow::Borrowed(" | "),
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
}

impl<'a, const IL: usize, const OL: usize> fmt::Display for CubeMatrixDisplay<'a, IL, OL> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (input_ix, &input) in self.cube.input.iter().enumerate() {
            write!(f, "{}", self.format.char_for_input(input))?;
            if input_ix < IL - 1 {
                write!(f, "{}", self.internal_separator)?;
            }
        }

        if IL > 0 && OL > 0 {
            write!(f, "{}", self.input_output_separator)?;
        }

        for (output_ix, &output) in self.cube.output.iter().enumerate() {
            write!(f, "{}", self.format.char_for_output(output))?;
            if output_ix < OL - 1 {
                write!(f, "{}", self.internal_separator)?;
            }
        }

        Ok(())
    }
}

#[derive(Copy, Clone, Debug)]
pub enum MatrixDisplayFormat {
    /// Display a cover using the format `100-1 | 10`, with dashes representing the
    /// don't care set.
    Dashes,

    /// Display a cover using the format `10021 | 43`, with numeric identifiers
    /// representing the don't care set.
    Numeric,
}

impl MatrixDisplayFormat {
    /// Returns the character that would be displayed for an input.
    pub fn char_for_input(self, input: Option<bool>) -> char {
        match input {
            Some(true) => '1',
            Some(false) => '0',
            None => match self {
                Self::Dashes => '-',
                Self::Numeric => '2',
            },
        }
    }

    /// Returns the character that would be displayed for an output.
    pub fn char_for_output(self, output: bool) -> char {
        match (self, output) {
            (Self::Dashes, true) => '1',
            (Self::Dashes, false) => '0',
            (Self::Numeric, true) => '4',
            (Self::Numeric, false) => '3',
        }
    }
}

impl Default for MatrixDisplayFormat {
    fn default() -> Self {
        Self::Dashes
    }
}

pub struct CubeAlgebraicDisplay<'a, const IL: usize, const OL: usize> {
    cube: &'a Cube<IL, OL>,
}

impl<'a, const IL: usize, const OL: usize> CubeAlgebraicDisplay<'a, IL, OL> {
    pub fn new(cube: &'a Cube<IL, OL>) -> Self {
        Self { cube }
    }
}

impl<'a, const IL: usize, const OL: usize> fmt::Display for CubeAlgebraicDisplay<'a, IL, OL> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if OL != 0 {
            for output_ix in 0..OL {
                match self.cube.output[output_ix] {
                    true => write!(f, "{}", AlgebraicSymbol::output(output_ix))?,
                    false => write!(f, "{}'", AlgebraicSymbol::output(output_ix))?,
                }
            }

            write!(f, " = ")?;
        }

        for input_ix in 0..IL {
            match self.cube.input[input_ix] {
                Some(true) => write!(f, "{}", AlgebraicSymbol::input(input_ix))?,
                Some(false) => write!(f, "{}'", AlgebraicSymbol::input(input_ix))?,
                None => {}
            };
        }

        Ok(())
    }
}

const INPUT_ALGEBRAIC_SYMBOLS: [char; 26] = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
    't', 'u', 'v', 'w', 'x', 'y', 'z',
];

const OUTPUT_ALGEBRAIC_SYMBOLS: [char; 26] = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
    'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
];

#[derive(Debug)]
pub(crate) enum AlgebraicSymbol {
    Char(char),
    String(String),
}

impl AlgebraicSymbol {
    #[inline]
    pub(crate) fn input(input_ix: usize) -> Self {
        Self::compute(input_ix, &INPUT_ALGEBRAIC_SYMBOLS)
    }

    #[inline]
    pub(crate) fn output(output_ix: usize) -> Self {
        Self::compute(output_ix, &OUTPUT_ALGEBRAIC_SYMBOLS)
    }

    fn compute(ix: usize, table: &[char; 26]) -> Self {
        if ix < 26 {
            return Self::Char(table[ix]);
        }
        let rest = ix / 26;
        let last = ix % 26;
        let last_ch = INPUT_ALGEBRAIC_SYMBOLS[last];

        match Self::input(rest) {
            Self::Char(ch) => Self::String(format!("{}{}", ch, last_ch)),
            Self::String(mut s) => {
                s.push(last_ch);
                Self::String(s)
            }
        }
    }
}

impl fmt::Display for AlgebraicSymbol {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Char(ch) => write!(f, "{}", *ch),
            Self::String(s) => write!(f, "{}", s),
        }
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
}
