#!/bin/sh
# Source manifest for testdata/niagara/ (git-ignored, never committed or
# redistributed): free Unreal Engine 5 editor (uncooked) Niagara systems used
# to test crates/fx_import locally. Run from the repository root:
#   sh testdata/niagara_assets.sh
# Tests that read these files skip when they are absent.
#
# File                         Engine  Source (GitHub repo @ commit, path)                         Licence
# NS_Fire.uasset               5.4     Josephy5/Project-Titan-Submissions@3c65d02 Fire/NS_Fire       MIT
# NS_FireSmokeOnly.uasset      5.4     Josephy5/Project-Titan-Submissions Fire/NS_FireSmokeOnly      MIT
# NS_ShinyGlimmer.uasset       5.4     Josephy5/Project-Titan-Submissions ShinyGlimmer/NS_ShinyGlimmer MIT
# NS_GeyserEruption.uasset     5.4     Josephy5/Project-Titan-Submissions FX_GeyserEruption/Geyser/  MIT
# NS_Fish.uasset               5.4     Josephy5/Project-Titan-Submissions Fish/NS_Fish               MIT
# NS_Boids.uasset              5.5     Shadertech/UE5NiagaraComputeShaderIntegration@db0cd21          MIT
#                                      Content/Niagara/NS_Boids
# NS_FluidSimulation.uasset    5.1     mushe/NiagaraFluid@f36c55b Content/VFXCollections/            MIT
#                                      FluidSimulation/FluidSimulationSystem (renamed)
# NS_DestructibleRig.uasset    5.5     eanticev/niagara-destruction-driver@881c350                   MIT
#                                      Content/PS_DestructibleRig (renamed)
#
# Note: these systems reference modules, materials and meshes in other
# packages of their projects; only the system packages are fetched.
set -e
D=testdata/niagara
mkdir -p $D
gh() { curl -sfL -o "$D/$3" "https://raw.githubusercontent.com/$1/$2"; }
T=Josephy5/Project-Titan-Submissions/3c65d02ac756e1ce9aa5b49ec984cd37e6c1af7d
gh $T Fire/NS_Fire.uasset NS_Fire.uasset
gh $T Fire/NS_FireSmokeOnly.uasset NS_FireSmokeOnly.uasset
gh $T ShinyGlimmer/NS_ShinyGlimmer.uasset NS_ShinyGlimmer.uasset
gh $T FX_GeyserEruption/Geyser/NS_GeyserEruption.uasset NS_GeyserEruption.uasset
gh $T Fish/NS_Fish.uasset NS_Fish.uasset
gh Shadertech/UE5NiagaraComputeShaderIntegration/db0cd215d8fb17d558bf1d81e6e22b67750dfdb7 \
   Content/Niagara/NS_Boids.uasset NS_Boids.uasset
gh mushe/NiagaraFluid/f36c55b0383418983ca1d46067669c2b477706df \
   Content/VFXCollections/FluidSimulation/FluidSimulationSystem.uasset NS_FluidSimulation.uasset
gh eanticev/niagara-destruction-driver/881c350b0670a3a99dfe240ffdf284e1df6d9a86 \
   Content/PS_DestructibleRig.uasset NS_DestructibleRig.uasset
(cd $D && sha256sum -c --quiet) <<'SUMS'
c42164fdfc001a8d7906cdfc1c4c1740f17675f4ddd84cea940d9f1538add8cc  NS_Boids.uasset
6a07737b17c6f3a9d89f16b46d57e260b8912e40ca81c95052d2222f553b6ed2  NS_DestructibleRig.uasset
0aaab687224dc45e499631c1bf9ed735d900ca34430fc38f5c50cd9c2931d39e  NS_FireSmokeOnly.uasset
500015aec992e62678ce3d1d16096cbd1725dfc0a53af0a8636adcc1d760aca3  NS_Fire.uasset
3ea1ae102ab3365a17ef859812abed97a7af84660ec28d8d9a80d40d5e7990fd  NS_Fish.uasset
1f22ecd7a3494a640e952ee85819ee6233e87306aa68d67e7ac271b4797b07f7  NS_FluidSimulation.uasset
17d36afb36d6d878364e7d9ae143e6469435fa82ecd57baff6dffc59bde90243  NS_GeyserEruption.uasset
d119d4412094d1c214e315c335ae84d9288b237491b552803b081dee02cda9ab  NS_ShinyGlimmer.uasset
SUMS
echo "fetched into $D"
