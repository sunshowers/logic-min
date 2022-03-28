// Copyright (c) The logic-min Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use clap::Parser;
use color_eyre::Result;
use fixture_details::AllFixtures;

#[derive(Debug, Parser)]
pub struct FixtureManagerApp {
    #[clap(subcommand)]
    command: FixtureManagerCommand,
}

#[derive(Debug, Parser)]
pub enum FixtureManagerCommand {
    GenerateInputs {
        #[clap(long, short, default_value_t = 64)]
        count: usize,
    },
    GenerateOutputs,
}

impl FixtureManagerApp {
    pub fn exec(self) -> Result<()> {
        self.command.exec()
    }
}

impl FixtureManagerCommand {
    pub fn exec(self) -> Result<()> {
        match self {
            Self::GenerateInputs { count } => {
                AllFixtures::generate_8_4(count)?;
                Ok(())
            }
            Self::GenerateOutputs => {
                unimplemented!("need to implement generate-outputs");
            }
        }
    }
}
