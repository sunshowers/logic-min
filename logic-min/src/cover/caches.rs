// Copyright (c) The logic-min Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::cube::Cube;
use once_cell::sync::OnceCell;
use std::{collections::BTreeSet, marker::PhantomData};

/// Cache for cover data.
#[derive(Clone, Debug, Default)]
pub(super) struct CoverCache<const IL: usize, const OL: usize> {
    unate_data: OnceCell<UnateData<IL>>,
    _marker: PhantomData<[(); OL]>,
}

impl<const IL: usize, const OL: usize> CoverCache<IL, OL> {
    pub(super) fn invalidate(&mut self) {
        self.unate_data = OnceCell::new();
    }

    pub(super) fn get_or_init_unate_data(
        &self,
        elements: &BTreeSet<Cube<IL, OL>>,
    ) -> &UnateData<IL> {
        self.unate_data.get_or_init(|| UnateData::new(elements))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct UnateData<const IL: usize> {
    pub(super) data: [ColumnData; IL],
    pub(super) meaningful_input_count: usize,
    pub(super) binate_input_count: usize,
}

impl<const IL: usize> UnateData<IL> {
    fn new<const OL: usize>(elements: &BTreeSet<Cube<IL, OL>>) -> Self {
        let mut data = [ColumnData::AllNones; IL];

        if elements.len() == 0 {
            return Self {
                data,
                meaningful_input_count: 0,
                binate_input_count: 0,
            };
        }

        let mut elements_iter = elements.iter();
        let first_cube = elements_iter
            .next()
            .expect("just checked that elements.len != 0");
        for input_ix in 0..IL {
            data[input_ix] = match first_cube.input[input_ix] {
                Some(true) => ColumnData::AllTrue,
                Some(false) => ColumnData::AllFalse,
                None => ColumnData::AllNones,
            };
        }

        let mut binate_input_count = 0;

        for element in elements {
            for input_ix in 0..IL {
                data[input_ix] = match (data[input_ix], element.input[input_ix]) {
                    (ColumnData::Binate, _) => ColumnData::Binate,
                    (ColumnData::AllTrue, Some(true)) => ColumnData::AllTrue,
                    (ColumnData::AllNones | ColumnData::TrueOrNone, Some(true)) => {
                        ColumnData::TrueOrNone
                    }
                    (ColumnData::AllFalse | ColumnData::FalseOrNone, Some(true)) => {
                        binate_input_count += 1;
                        ColumnData::Binate
                    }
                    (ColumnData::AllNones, None) => ColumnData::AllNones,
                    (ColumnData::AllTrue | ColumnData::TrueOrNone, None) => ColumnData::TrueOrNone,
                    (ColumnData::AllFalse | ColumnData::FalseOrNone, None) => {
                        ColumnData::FalseOrNone
                    }
                    (ColumnData::AllFalse, Some(false)) => ColumnData::AllFalse,
                    (ColumnData::AllNones | ColumnData::FalseOrNone, Some(false)) => {
                        ColumnData::FalseOrNone
                    }
                    (ColumnData::AllTrue | ColumnData::TrueOrNone, Some(false)) => {
                        binate_input_count += 1;
                        ColumnData::Binate
                    }
                };
            }
        }

        let meaningful_input_count = data.iter().filter(|c| **c != ColumnData::AllNones).count();
        Self {
            data,
            meaningful_input_count,
            binate_input_count,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(super) enum ColumnData {
    /// All None elements (or the cover is empty)
    AllNones,
    /// All false
    AllFalse,
    /// All true
    AllTrue,
    /// Negative unate not covered by one of the above
    FalseOrNone,
    /// Positive unate not covered by one of the above
    TrueOrNone,
    /// Binate column
    Binate,
}

impl ColumnData {
    #[inline]
    pub(super) fn is_meaningful(self) -> bool {
        !matches!(self, Self::AllNones)
    }

    #[inline]
    pub(super) fn is_monotone_increasing(self) -> bool {
        matches!(self, Self::AllNones | Self::AllTrue | Self::TrueOrNone)
    }

    #[inline]
    pub(super) fn is_monotone_decreasing(self) -> bool {
        matches!(self, Self::AllNones | Self::AllFalse | Self::FalseOrNone)
    }

    #[inline]
    pub(super) fn is_unate(self) -> bool {
        !matches!(self, Self::Binate)
    }
}
