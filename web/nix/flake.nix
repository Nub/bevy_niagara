{
  description = "animatey web build: Rust with the wasm32 target, wasm-bindgen-cli matching Cargo.lock";

  # Same nixpkgs and rust-overlay revisions as the bevy-solari-opt fork's
  # flake.lock, so rustc is the fork's (stable.latest at that overlay).
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/31ecbaca1bdefa45dffc409f0250bd8039ea0010";
    rust-overlay = {
      url = "github:oxalica/rust-overlay/5aa410a1df0e4e9c25f3818320651c2bddf66aff";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {nixpkgs, rust-overlay, ...}: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      overlays = [rust-overlay.overlays.default];
    };
    rust = pkgs.rust-bin.stable.latest.default.override {targets = ["wasm32-unknown-unknown"];};
    # wasm-bindgen-cli must equal the wasm-bindgen version in Cargo.lock (0.2.130).
    wasm-bindgen-cli = pkgs.buildWasmBindgenCli rec {
      src = pkgs.fetchCrate {
        pname = "wasm-bindgen-cli";
        version = "0.2.130";
        hash = "sha256-9Og1O/6/VC3sc5hBjoIWBErmLW2UHGMsTQO+6+m/dng=";
      };
      cargoDeps = pkgs.rustPlatform.fetchCargoVendor {
        inherit src;
        inherit (src) pname version;
        hash = "sha256-j81sUf1m5yXU4FJfGj7GV+hdqVCV8mRmNb0QIsMCjqw=";
      };
    };
  in {
    devShells.${system}.default = pkgs.mkShell {
      nativeBuildInputs = [rust wasm-bindgen-cli pkgs.binaryen pkgs.lld];
    };
  };
}
