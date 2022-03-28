// Copyright (c) The logic-min Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::value_generator::ValueGenerator;
use camino::Utf8PathBuf;
use color_eyre::Result;
use logic_min::cover::Cover;
use once_cell::sync::Lazy;
use proptest::prelude::*;

pub struct AllFixtures {
    _dir: Utf8PathBuf,
}

static ALL_FIXTURES_STATIC: Lazy<AllFixtures> = Lazy::new(AllFixtures::init);

impl AllFixtures {
    pub fn get() -> &'static Self {
        &*ALL_FIXTURES_STATIC
    }

    fn init() -> Self {
        let dir: Utf8PathBuf = env!("CARGO_MANIFEST_DIR").into();
        let dir = dir.parent().unwrap().join("data");
        Self { _dir: dir }
    }

    pub fn generate_8_4(count: usize) -> Result<()> {
        let mut value_gen = ValueGenerator::from_seed("logic-min_8_4");

        let mut tautology_count = 0;
        for _ in 0..count {
            let mut gen = value_gen.partial_clone();
            let cover = gen.generate(any::<Cover<8, 4>>());
            if cover.is_tautology() {
                tautology_count += 1;
            }
        }

        println!("tautology count: {}", tautology_count);

        Ok(())
    }
}
