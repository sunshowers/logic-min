// Copyright (c) The logic-min Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    cover::Cover,
    cube::{AlgebraicSymbol, Cube, MatrixDisplayFormat},
};
use itertools::{Itertools, Position};
use std::{borrow::Cow, cmp::Ordering, fmt};

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
