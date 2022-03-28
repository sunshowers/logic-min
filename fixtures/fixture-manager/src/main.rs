// Copyright (c) The logic-min Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use clap::Parser;
use color_eyre::Result;
use fixture_manager::FixtureManagerApp;

fn main() -> Result<()> {
    let app = FixtureManagerApp::parse();
    app.exec()
}
