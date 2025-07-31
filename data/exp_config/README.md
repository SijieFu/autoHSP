# Experiment Configuration
- This folder contains `.toml` configurations for running experiments for specific test materials in the lab.
- The `default.toml` file is the default configuration file.
- You **MUST** create a `.toml` file for a resin you intend to test with the file name `{resin_code}.toml` in this directory before running experiments. For example, `R1.toml` for resin with the code name `R1` in the `resins.csv` file. This is to ensure that you have considered any incompatible solvents with the specific resin for the sake of lab personnel.
- `{resin_code}.toml` overwrites the default `default.toml` to generate a final configuration.