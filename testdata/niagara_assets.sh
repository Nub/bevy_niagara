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
# Dependency packages (materials, textures, meshes, module scripts) go to
# testdata/niagara/packages/<mount>/<path>.uasset, laid out like the
# projects' mount points (/Game/..., /ST_NiagaraCSIntegration/...,
# /NiagaraDestructionDriver/...), where `fximport convert` looks for them.
# Not in the source repositories (listed as missing by the converter):
# T_CausticSmoke, T_LavaMolten_D (fire material), SM_SmokeParticle and
# MI_GeyserSmoke_VFX (geyser), DefaultSpriteMaterial_DEBUG (destruction rig).
#
# Unreal reference media (testdata/niagara/reference/, from each repo's
# README): Project Titan videos (editor viewport recordings), the Boids and
# fluid GIFs, the destruction driver's demo.webp.
set -e
D=testdata/niagara
mkdir -p $D/packages $D/reference
gh() { curl -sfL -o "$D/$3" "https://raw.githubusercontent.com/$1/$2"; }
# pkg REPO@COMMIT REPO_PATH MOUNT_PATH (both without .uasset)
pkg() { mkdir -p "$D/packages/$(dirname "$3")"; curl -sfL -o "$D/packages/$3.uasset" "https://raw.githubusercontent.com/$1/$2.uasset"; }
ref() { curl -sfL -o "$D/reference/$2" "$1"; }
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
G=Game/VFX/Environment
for n in MI_FireParticles2Instance MI_FireParticlesInstance MI_HeatDistortInstance M_FireParticles2MaskBase M_FireParticlesMaskBase M_HeatDistort; do
  pkg $T Fire/Materials/$n $G/_Global/Fire/Materials/$n
done
for n in T_CausticNoise T_FireParticles2Sprite T_FireParticlesSprite; do pkg $T Fire/Texture/$n $G/_Global/Fire/Texture/$n; done
for n in MI_FishExample MI_FishExample2 MI_FishExample3 M_FishExample M_FishExample2 M_FishExample3; do
  pkg $T Fish/Material/$n $G/Clifftop/Fish/Material/$n
done
for n in SM_FishExample SM_FishExample2 SM_FishExample3; do pkg $T Fish/Mesh/$n $G/Clifftop/Fish/Mesh/$n; done
for n in M_StarMat M_StarMatInst; do pkg $T ShinyGlimmer/Materials/$n $G/Clifftop/ShinyGlimmer/Materials/$n; done
for n in T_StarSprite T_StarSpriteHighRes; do pkg $T ShinyGlimmer/Texture/$n $G/Clifftop/ShinyGlimmer/Texture/$n; done
S=Shadertech/UE5NiagaraComputeShaderIntegration/db0cd215d8fb17d558bf1d81e6e22b67750dfdb7
for n in Materials/MI_Boid Materials/M_Boid Meshes/SM_Boid Niagara/Scripts/NM_Boids; do pkg $S Content/$n ST_NiagaraCSIntegration/$n; done
E=eanticev/niagara-destruction-driver/881c350b0670a3a99dfe240ffdf284e1df6d9a86
for n in Defaults/RT_DestructibleRig_SimulatedBone_Positions_DEBUG Defaults/RT_DestructibleRig_SimulatedBone_Rotations_DEBUG1 \
         Defaults/T_PreviewMesh_InitialBoneLocations Defaults/SM_PreviewMesh_NDD NiagaraModules/ApplyDetachementForce \
         NiagaraModules/WriteParticleTransformsToRenderTarget_Simple; do
  pkg $E Content/$n NiagaraDestructionDriver/$n
done
A=https://github.com/Josephy5/Project-Titan-Submissions/assets/59676928
ref $A/1dbdcb81-7777-46cb-8fe3-5725d1624ace NS_Fish.mp4
ref $A/31b05964-5f87-4f3d-aa1b-1027083397fe NS_ShinyGlimmer.mp4
ref $A/2f072d35-5434-4822-a257-9eea28922b5c NS_FireSmokeOnly.mp4
ref $A/9cdd299b-f512-4938-8859-74e8deddb3e1 NS_Fire.mp4
ref $A/eddff1d3-8009-4c17-90e3-b99f80d375d2 NS_GeyserEruption.mp4
ref https://github.com/user-attachments/assets/002d4802-f1b4-402e-b06f-34c67bafe741 NS_Boids.gif
ref https://user-images.githubusercontent.com/26865534/184529081-7178c81c-4608-4519-ba0b-40f59fb07af4.gif NS_FluidSimulation.gif
ref https://raw.githubusercontent.com/$E/Resources/demo.webp NS_DestructibleRig.webp
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
(cd $D/packages && sha256sum -c --quiet) <<'SUMS'
21401271f542f17a98df369727a448d20e45a8a0ff0ae2b3685b77cd900647af  Game/VFX/Environment/Clifftop/Fish/Material/M_FishExample2.uasset
c1bd77b389fbcefeb6e7e6c5d3be6e71df6c4208b521aad6556db42d7255bd01  Game/VFX/Environment/Clifftop/Fish/Material/M_FishExample3.uasset
53d1e6cc402087d1eb2b8391e30dae22edd638148ce2dba884382c62c7075a51  Game/VFX/Environment/Clifftop/Fish/Material/M_FishExample.uasset
60a61a7fa2f644c72f886d57d7d86cce1cdddceb29f01bc54bdf1969d12d949f  Game/VFX/Environment/Clifftop/Fish/Material/MI_FishExample2.uasset
00c5dc026bedad0dc631b8e46eb1a0d7f6f999cdb20563f9cf1cedd1be99b22d  Game/VFX/Environment/Clifftop/Fish/Material/MI_FishExample3.uasset
d0cd516acab78e5b41e0b3ae4824bda59bd511e776f5f983b0fde5d32d8a7e91  Game/VFX/Environment/Clifftop/Fish/Material/MI_FishExample.uasset
6e03cc1d4b8febed7cce456830b93b99d928181d82bfe6bbbc67288f2748dcb9  Game/VFX/Environment/Clifftop/Fish/Mesh/SM_FishExample2.uasset
14afcd42583d965e949ed4b33611906feacd9f52ff8c703edd3ad4bce48d760e  Game/VFX/Environment/Clifftop/Fish/Mesh/SM_FishExample3.uasset
4ace29d7e3e4a28a6d2594f36067e3aa123779ffac7e882e4dcac99effbb20cf  Game/VFX/Environment/Clifftop/Fish/Mesh/SM_FishExample.uasset
d58e701c5431305d1f89fed58f26c715d9c75a0e08624b0e5bc08b4bcdd4f75e  Game/VFX/Environment/Clifftop/ShinyGlimmer/Materials/M_StarMatInst.uasset
7f24fee231c7a667ec692bd49c9735f15fd1c0a1a0c525ea3887cbc0b3a989b8  Game/VFX/Environment/Clifftop/ShinyGlimmer/Materials/M_StarMat.uasset
d7e1cfbf6c1d94c9a481838741981babe1fc91b77fe5ea3caf79918e70182c55  Game/VFX/Environment/Clifftop/ShinyGlimmer/Texture/T_StarSpriteHighRes.uasset
ee633f572409589aebee82da9ab8603221bb855a86b875e7b41f5468b9077198  Game/VFX/Environment/Clifftop/ShinyGlimmer/Texture/T_StarSprite.uasset
b0c71c0fb8aaab98efc45ffd84188b97a9184ea4a20e67dadd71f643fe563531  Game/VFX/Environment/_Global/Fire/Materials/M_FireParticles2MaskBase.uasset
ef2b32e9a9833d98926522c9ae8eab8207a145d03904fabc1c6598e1938a5e52  Game/VFX/Environment/_Global/Fire/Materials/M_FireParticlesMaskBase.uasset
ac3dc1c7f4a573e86419089308c7242c839d6d2fc0681a45dbca6e368aaff17a  Game/VFX/Environment/_Global/Fire/Materials/M_HeatDistort.uasset
6cb66f1704552e77745012c1c8a101f570e99240e848d17bf58a0ae81fc13ae0  Game/VFX/Environment/_Global/Fire/Materials/MI_FireParticles2Instance.uasset
a20ee669fc5e37de9edcccc70ac54ee1ff02726409214ffa6543bb41d486e308  Game/VFX/Environment/_Global/Fire/Materials/MI_FireParticlesInstance.uasset
e67a621c8408e45548a2382c8bbe4e283ecec4b30d5e3f7e9565b5cd94de6263  Game/VFX/Environment/_Global/Fire/Materials/MI_HeatDistortInstance.uasset
caccfe94d5d8247985977cbc8a9d585d916306aed18b44fb9be161137aa1b7c5  Game/VFX/Environment/_Global/Fire/Texture/T_CausticNoise.uasset
475722b259fd3eec3b0784eb34169609acd927f029cee5492f608575282b5caf  Game/VFX/Environment/_Global/Fire/Texture/T_FireParticles2Sprite.uasset
248ced66f38dd34f5a63d62573642fe76882b4393b6a54c875d45c62967c01e6  Game/VFX/Environment/_Global/Fire/Texture/T_FireParticlesSprite.uasset
807e4334536dc5dfb80fe1b52391f7fe6407e4508a14b19be938e0147a492710  NiagaraDestructionDriver/Defaults/RT_DestructibleRig_SimulatedBone_Positions_DEBUG.uasset
9897a7a5aad01c7b9d3b44a1dc5488a10567a264b9254f700fbc59af66d676da  NiagaraDestructionDriver/Defaults/RT_DestructibleRig_SimulatedBone_Rotations_DEBUG1.uasset
8b2472e6a5fa102f8d4691be8234f4d8e33682952b6f4ae551d7d332ba7c2d95  NiagaraDestructionDriver/Defaults/SM_PreviewMesh_NDD.uasset
02b6432ed0412ba2f4b3fe0dadaff78eaa1d7e1efa614dfe248d4eb9c6059aeb  NiagaraDestructionDriver/Defaults/T_PreviewMesh_InitialBoneLocations.uasset
af6643df1fe153b94d68594021976590b11eddf1d4ca1d2180431230312fb423  NiagaraDestructionDriver/NiagaraModules/ApplyDetachementForce.uasset
7bc6b24094c592a6883512f79a33acac6f19f217afd3fa855a2451c0597ddb0b  NiagaraDestructionDriver/NiagaraModules/WriteParticleTransformsToRenderTarget_Simple.uasset
196e73a0eab7a6097312635fec4b97ddff3c118ca046b8353a572fa93af1e395  ST_NiagaraCSIntegration/Materials/M_Boid.uasset
f3817ae8c16ea1d5bd760cbc1ea0a23cfc5b28ac76c5908a23a3f02353e3fa98  ST_NiagaraCSIntegration/Materials/MI_Boid.uasset
7ed59b1cd7b1ab5a650447cd67465b44a1c9156b2c4d58b9d1efe6ee4a99cecc  ST_NiagaraCSIntegration/Meshes/SM_Boid.uasset
206b8ba19b208b5d653566ff2c8be58cd1bfdac0ce1920766f25570b21c789e4  ST_NiagaraCSIntegration/Niagara/Scripts/NM_Boids.uasset
SUMS
(cd $D/reference && sha256sum -c --quiet) <<'SUMS'
d9927bdceb8b3366050843dd0aea21084b45c61ddd4943954330aae62d1fb536  NS_Boids.gif
942886c3b463d2da130fa3b5d89ba217c142b6b8864768daa363e534d214b2ad  NS_DestructibleRig.webp
f687730996e30d76b4d664e2ee840258cc4a173cb6412a6629d77ed0c5ae84f3  NS_Fire.mp4
9e6fa4318727cfed51ab36f6141a5cad925dbb738e78d1d9bebae49d7a125c44  NS_FireSmokeOnly.mp4
298238fbe02d50b143244473b296d172a79be2d14277ef319ab4dbaa2ddfa80d  NS_Fish.mp4
d69d28927dac6bfd2a5fb1341c1208d368b2a045ee9612f4a96c2276a6c83b8e  NS_FluidSimulation.gif
fc2ea67c73ac0b425f7d91ba7a8ba6446780584f8483f033bdb73259a4a4eea4  NS_GeyserEruption.mp4
0a866476f47bd961f62e1158ec4ca4eed68dbe0335167607e55ca6d6fe51ea45  NS_ShinyGlimmer.mp4
SUMS
echo "fetched into $D"
