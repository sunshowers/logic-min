// Copyright (c) The logic-min Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    cover::Cover,
    cube::{AlgebraicSymbol, Cube, MatrixDisplayFormat},
};
use itertools::{Itertools, Position};
use std::{borrow::Cow, cmp::Ordering, fmt};

impl<const IL: usize, const OL: usize> fmt::Debug for Cover<IL, OL> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.try_as_cover0() {
            Some(cover0) => f
                .debug_tuple("Cover")
                .field(&format_args!("{}", cover0.algebraic_display()))
                .finish(),
            None => {
                let mut debug_struct = f.debug_struct("Cover");
                for output_ix in 0..OL {
                    let component = self.output_component(output_ix).collect();
                    debug_struct.field(
                        &format!("{}", AlgebraicSymbol::output(output_ix)),
                        &format_args!("{}", AlgebraicDisplay0::new(component)),
                    );
                }
                // Also print out information for cubes that aren't in any output components.
                let match_none: Vec<_> = self
                    .elements()
                    .iter()
                    .filter_map(|elem| {
                        elem.output
                            .iter()
                            .all(|v| !*v)
                            .then(|| elem.as_input_cube())
                    })
                    .collect();
                if !match_none.is_empty() {
                    debug_struct.field(
                        "(no outputs)",
                        &format_args!("{}", AlgebraicDisplay0::new(match_none)),
                    );
                }

                debug_struct.finish()
            }
        }
    }
}

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
        for (elem_ix, elem) in self.cover.elements().iter().enumerate() {
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
            Some(cover0) => write!(
                f,
                "{}",
                AlgebraicDisplay0::new(cover0.elements().iter().collect())
            ),
            None => {
                let (separator, print_last) = &self.separator;
                // For each output value, print out the corresponding cubes in the component.
                for output_ix in 0..OL {
                    write!(
                        f,
                        "{} = {}",
                        AlgebraicSymbol::output(output_ix),
                        AlgebraicDisplay0::new(self.cover.output_component(output_ix).collect())
                    )?;
                    if output_ix < OL - 1 || *print_last {
                        write!(f, "{}", separator)?;
                    }
                }
                Ok(())
            }
        }
    }
}

struct AlgebraicDisplay0<'a, const IL: usize> {
    elements: Vec<&'a Cube<IL, 0>>,
}

impl<'a, const IL: usize> AlgebraicDisplay0<'a, IL> {
    fn new(mut elements: Vec<&'a Cube<IL, 0>>) -> Self {
        // Sort the elements lexicographically in the order [Some(true), Some(false), None],
        // This results in minterms starting with `a` showing up first, then `a'`, then
        // minterms not containing a.
        elements.sort_unstable_by(|a, b| {
            for input_ix in 0..IL {
                match (a.input[input_ix], b.input[input_ix]) {
                    (Some(true), Some(true)) | (Some(false), Some(false)) | (None, None) => {
                        continue
                    }
                    (Some(true), Some(false) | None) => return Ordering::Less,
                    (Some(false) | None, Some(true)) => return Ordering::Greater,
                    (Some(false), None) => return Ordering::Less,
                    (None, Some(false)) => return Ordering::Greater,
                }
            }
            Ordering::Equal
        });

        Self { elements }
    }
}

impl<'a, const IL: usize> fmt::Display for AlgebraicDisplay0<'a, IL> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.elements.is_empty() {
            return write!(f, "(none)");
        }
        for elem in self.elements.iter().with_position() {
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
}
