# bevy_niagara

A Niagara-style GPU particle system for [Bevy](https://bevyengine.org), with an importer for Unreal Niagara `.uasset` files.

- **fx_core**: the effect format (`.fx.ron`): systems, emitters, Niagara modules, the `fxc` checker and cooker
- **fx_gpu** / **fx_render**: GPU simulation, sorting and batched rendering (sprites, ribbons, meshes, lights, decals, volumes, heat-haze distortion)
- **fx_solari**: the Solari ray-traced path
- **fx_import**: reads Niagara `.uasset` packages (UE 4.27 and 5.x) and converts them to `.fx.ron`, with materials, textures and meshes (`fximport`)
- **fx_preview**: headless renders and videos (`fxpreview`) and GPU benchmarks (`fxbench`)
- **fx_viewer**, **fx_mcp**, **fx_bench**: live viewer, MCP server for authoring effects, benchmark suite

Effects live in [`effects/`](effects/README.md); benchmarks in [`bench/`](bench/README.md).

## Building

The workspace builds against a Bevy 0.20-dev fork (`bevy-solari-opt`) that is referenced by local path in `Cargo.toml`; it does not build without that checkout.

Third-party Niagara files used for testing are downloaded by the scripts in `testdata/` into git-ignored folders and are not part of this repository.

Unreal, Unreal Engine and Niagara are trademarks of Epic Games, Inc. This project is not affiliated with or endorsed by Epic Games.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option. `effects/samples/models/Fox.glb` comes from the Bevy repository under its own license.
