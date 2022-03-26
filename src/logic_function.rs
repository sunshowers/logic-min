// Copyright (c) The logic-min Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::cover::Cover;

#[derive(Clone, Debug)]
pub struct LogicFunction<const IL: usize, const OL: usize> {
    pub on_set: Cover<IL, OL>,
    pub dc_set: Cover<IL, OL>,
}
