// Copyright (c) The logic-min Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{cover::Cover, cube::Cube};
use arrayvec::ArrayVec;
use proptest::prelude::*;
use std::fmt;

impl<const IL: usize, const OL: usize> Arbitrary for Cube<IL, OL> {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
        // Generate IL input values and OL output values.
        let input_strategy = prop::collection::vec(any::<Option<bool>>(), IL);
        let output_strategy = prop::collection::vec(any::<bool>(), OL);
        (input_strategy, output_strategy)
            .prop_map(|(input_vec, output_vec)| {
                let input = vec_to_array(input_vec);
                let output = vec_to_array(output_vec);
                Self { input, output }
            })
            .boxed()
    }
}

impl<const IL: usize, const OL: usize> Arbitrary for Cover<IL, OL> {
    type Parameters = Option<(usize, usize)>;
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(params: Self::Parameters) -> Self::Strategy {
        let (min_size, max_size) = params.unwrap_or((0, IL * OL * IL));
        // Generate somewhere between min_size and max_size cubes.
        prop::collection::btree_set(any::<Cube<IL, OL>>(), min_size..max_size)
            .prop_map(|elements| Self::new(elements))
            .boxed()
    }
}

#[inline]
fn vec_to_array<T: fmt::Debug, const N: usize>(vec: Vec<T>) -> [T; N] {
    let array_vec: ArrayVec<T, N> = vec.into_iter().collect();
    array_vec
        .into_inner()
        .expect("vec should be exactly N elements long")
}
