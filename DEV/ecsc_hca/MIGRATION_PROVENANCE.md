# ECS_C / HCA R0–R19 — Migration Provenance

## Authority and limitations

- Source workspace: `/mnt/primesauce/Elpis_DEV_WORK/ecsc_hca_r0`, branch `research/ecsc_hca_r0`.
- Historical underlying runtime commit: `cb530e7a0c290b9ceeb6614ab1db4ab45ada54e3`.
- This commit pins the runtime, **not** the untracked R0–R19 research directory.
- Historical `src/elpis/runtime/context.py` SHA-256: `afa96f35c51e8130683eb5502e8e6478d7c113caf1f6116e4a703c6cd0aae2f0`.
- Frozen `BASELINE_R1_FREEZE.json` SHA-256: `a5d136abab626f1887a67fbf7bc887ad17e855ad5c2ecf820ca6f1ff97156c10`.
- Historical complete pytest suite: reported PASS, exit code 0, using `PYTHONPATH=src:.:research/ecsc_hca_r0` in original research workspace.
- Destination `elpis.runtime` is different; destination runtime compatibility is **not qualified**. No historical runtime modules were copied into canonical `src/elpis/runtime/`.
- No production integration into `src/elpis/ECS_C/` is claimed.
- R19 disposition: `DEV_DESCRIPTIVE_NOT_SCIENTIFIC_QUAL`; independent leakage, shortcut, split and replay audit remains outstanding. R20 is not authorized.
- Bulk external evidence, datasets, checkpoints, caches, and unrelated `HECS_R3/` and `evidence/` are excluded.
- This manifest authorizes no Git staging, commit, branch change or push.

## Historical regression invocation

`PYTHONPATH=src:.:research/ecsc_hca_r0 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest research/ecsc_hca_r0 -q -p no:cacheprovider --tb=short --maxfail=10`

## Migrated file integrity

63 Python, 1 JSON and 23 Markdown files; all 87 verified byte-for-byte against original workspace at manifest creation.

| Destination-relative file | SHA-256 |
|---|---|
| `DEV/ecsc_hca/AGENT_TASK.md` | `df53f1089e34e66441f2897b39bea2269b493570295223a19b56caee9cba85ad` |
| `DEV/ecsc_hca/DRAFT_PROTOCOL_R1.md` | `358bb68101bc0fea3ad8ebd6ceffa70fa2f507ec120b0b265633508627692424` |
| `DEV/ecsc_hca/R10_PROTOCOL.md` | `2fc52208ab202cb298c896ee6f665464da0463d0d8061cec1bb3c1744eb2d27b` |
| `DEV/ecsc_hca/R11_PAIRED_AUDIT_PROTOCOL.md` | `e3cefa3bf46f10c9961ffcca10a70cc17a5ffe05da7d6d6b1e9e48cee197731f` |
| `DEV/ecsc_hca/R11_PROTOCOL.md` | `7afac8f97ecb572fca897c655c69a6247779a42a1e63e42b61b1964605d3c309` |
| `DEV/ecsc_hca/R12N_PROTOCOL.md` | `0a23e83fbecf16ad057a548b778536a475a4131354ece86d1cb879a89f136fe5` |
| `DEV/ecsc_hca/R12_PROTOCOL.md` | `1e550029c29e1459bf2852eb8133ad10ffd8ffd33c8018f0b5f93bc244708589` |
| `DEV/ecsc_hca/R13_PROTOCOL.md` | `ca2a7466ec575fabdd4e4a5113c3530c57e8a58c960b222074ec38dafb0ffdfb` |
| `DEV/ecsc_hca/R14_PROTOCOL.md` | `18b5aa96037f5f7af812f88d1a8f23ddedeb3004ea751ec0f7adfbdd8e574ada` |
| `DEV/ecsc_hca/R15_PROTOCOL.md` | `05e61fe57f59dde940584564525b4f98cedfcf2c99a8a80e4c9f01d0c76f3021` |
| `DEV/ecsc_hca/R16_PROTOCOL.md` | `8078955feb31e0b27c3035516dd615f2192543624d56213345b5c2c04537a0b2` |
| `DEV/ecsc_hca/R17_PROTOCOL.md` | `6d2f158cdd3aba5bb8c757c9787cd7e798ecb97d374ae403e9fe0138174aea35` |
| `DEV/ecsc_hca/R17_SCOPE_AUDIT_PROTOCOL.md` | `465cc7855f61e834582bc3c584a476a0951677dd4df6b24c5291cdf2a6d0472a` |
| `DEV/ecsc_hca/R18_PROTOCOL.md` | `14e768fc0f12f9da7db6332a650e7536bae01ae0099ce9c9d7f831ac85cdbc56` |
| `DEV/ecsc_hca/R18_SHORTCUT_AUDIT_PROTOCOL.md` | `94e3507cc76da2f924fb4a103ee7028d7f15abfe7cf3b8fa186cee09737ccbef` |
| `DEV/ecsc_hca/R19_PROTOCOL.md` | `84d70f4a5567dddb13b853fcc4b4a4faeeded7fe811b1f4df8c537483d61df1c` |
| `DEV/ecsc_hca/R2_CONTEXT_LIFETIME.md` | `b35ff0c5b0290b8bb49b9378c3cd1ba39d7a258ef0193ecb97a3bf784ed205b2` |
| `DEV/ecsc_hca/R2_MECHANICS_REPORT.md` | `92f12f1cdcc35a2c9137ec000b05dee20027997636660a06a77204b2125faf61` |
| `DEV/ecsc_hca/R3_PROTOCOL_DRAFT.md` | `78d7105962effc7b6b043c8833de6ab1149c1f52a0c41f931c21452f75b9d81c` |
| `DEV/ecsc_hca/R8_PROTOCOL.md` | `744fbc644c28431ef126072371699801b3144a163e4157fbd64c590a94effffe` |
| `DEV/ecsc_hca/R9_EXTERNAL_INTAKE_PROTOCOL.md` | `7542be9093a88c1fb2fb2fec9c9a1f7cdb590673f59922530442e44420869a07` |
| `DEV/ecsc_hca/README.md` | `47f3dfb9c5679d86d847f61cb5380ee69ba097d89dd0ee4addb80d77e38f43cf` |
| `DEV/ecsc_hca/RESEARCH_CONTRACT.md` | `7fee8e06dd230a6751a45993f499dc1c39a65d8064423ef82eb49f2c4c444af2` |
| `research/ecsc_hca_r0/BASELINE_R1_FREEZE.json` | `a5d136abab626f1887a67fbf7bc887ad17e855ad5c2ecf820ca6f1ff97156c10` |
| `research/ecsc_hca_r0/__init__.py` | `17225e72b52e2c053f7c7a5499fccbebc546e24cb0dad6baa56cb106dc0eed2c` |
| `research/ecsc_hca_r0/b2.py` | `cd49a8e8b55d908858728911fda0f31cf51fc5d5cf990b20ad01475fdc2c2aca` |
| `research/ecsc_hca_r0/baselines.py` | `04719f9c446f83b9de0b0d157f55e29f2aaa2ebfcee80a75272a5fe4fb8aef7a` |
| `research/ecsc_hca_r0/diagnose_r2_drift.py` | `140c0ded6b6c82ecdedaff5a49fb2933b49320b1d536265e68a28275002d11d3` |
| `research/ecsc_hca_r0/evaluation.py` | `b8901ab3a43d704df4eda08559013d89584d99b905055b5c0770900e1bed7462` |
| `research/ecsc_hca_r0/fixtures.py` | `0efe462aa16caf6838ee06bd1487aae6cc3470f436a9dcfc972dc386e70c5d16` |
| `research/ecsc_hca_r0/r10_blind_retrieval.py` | `b0ec92f28b4e28be71cd85426c81ea5a186df3b144f8fea5f6a44db9d1b6cf3e` |
| `research/ecsc_hca_r0/r11_candidate_graph.py` | `ef6e910f5ddd50cc818e56acfaf63fe607e906001a5d55c637d3d0444e6fe7c1` |
| `research/ecsc_hca_r0/r11_paired_audit.py` | `a4408d66f1f128c595f5349163cd1f450bbadcb9bc34966652c252b1a2ec57da` |
| `research/ecsc_hca_r0/r12_answer_level.py` | `40e22d1b1164efc7e1b498f5e49e4c6795f507fd615c4e15a8cff1ee49f9a83e` |
| `research/ecsc_hca_r0/r12_native_structure.py` | `3710a67b9f15f2a1435714f8209cad266f341cf2e1a274d9be05ca3bda2e5e03` |
| `research/ecsc_hca_r0/r13_ingress_null.py` | `b650c33bb1cf49fe70a3dc76ca3761e740d7384ef4348d4c02496010ff4a392e` |
| `research/ecsc_hca_r0/r14_native_projector.py` | `13dea8746cce926e3658db097bdb9396fe2333dcd2d5771e89cd0bc75c3f4591` |
| `research/ecsc_hca_r0/r15_candidate_ingress.py` | `a3de032c43d96714d13273677312da0ceaa6263a5a27ba16059f566626443950` |
| `research/ecsc_hca_r0/r16_scope_learning.py` | `9756cee23a064741289fcb0a73f16cf5f88a5f0395bb354d70be5872aed10aab` |
| `research/ecsc_hca_r0/r17_raw_scope.py` | `19a73bd44c05e253084a203f7f23174799e9e7601644b084e31c34089dfe2d05` |
| `research/ecsc_hca_r0/r17_scope_ablation_audit.py` | `73883b2c47c8ebdf6688293d96f8c5e78cac95c270950ae223201b7009513529` |
| `research/ecsc_hca_r0/r18_bracket_binding.py` | `14173dd48bd9596da67f3f83620d4ef6a336564d1bc2a9547da9eb6945a72a4d` |
| `research/ecsc_hca_r0/r18_shortcut_audit.py` | `2182fe1f8c8ec1b163a1b54c459c586a1cf33346b18e68377a0eabdd3f125859` |
| `research/ecsc_hca_r0/r19_fixed_root.py` | `c62207638f5dd8299c0269cad15b5d06c33b36b971b185b6a48598e2a8d141aa` |
| `research/ecsc_hca_r0/r2_b0i.py` | `863ee397ec3ffee502c61ba65877a34a37598b175672823f5c06e69a131d2f3a` |
| `research/ecsc_hca_r0/r2_b2s.py` | `f85339da85f21581eef3bf0b0f60f1fbaccf8c330473af10d1ef6181518c4b2d` |
| `research/ecsc_hca_r0/r2_common.py` | `8362855e08017403ab9ad93a9c88f60e5952d24898a3c1cac690f31df45e1d18` |
| `research/ecsc_hca_r0/r2_run.py` | `d3795cbd34b3d9aa285bf3bbbf3c92d3a010d14760aa8bbca7832142e7ef899e` |
| `research/ecsc_hca_r0/r2_trajectories.py` | `e7b285a61a0381c8ae9c3ec07025ba6b7a52cffdaa4f739f1a0a540368023c8f` |
| `research/ecsc_hca_r0/r3_warm_exact.py` | `a9934ffc9aff3e2a83f04f1a7a269048797c4d46e34dfb6a71239dac0f7ddc3f` |
| `research/ecsc_hca_r0/r4_hacf_sources.py` | `816faf8c91cb486c2987d9039878f36fbd111f08d0e5c11d483fed5678be2267` |
| `research/ecsc_hca_r0/r5_warm_publication.py` | `b2330a5822a2502734c182fdd9bf86e04149228d60be571c318bbdccbfd126ff` |
| `research/ecsc_hca_r0/r6_utility_probe.py` | `998670167b2b57cf5c6b710cc4be534aba7097810f2a724a37ef9c7daa3024bd` |
| `research/ecsc_hca_r0/r7_task_bound.py` | `65730c036ce56211d6fc1b56d7fe105da54bb9b6a30a268da0bbbc041207b332` |
| `research/ecsc_hca_r0/r8_accounting.py` | `c93972aa9e261f9a85057689e169d6d35b08336db9a0356a1352a59a2ab82cce` |
| `research/ecsc_hca_r0/r8_falsification.py` | `0a1a28ecc1b88b19200fdb9ea787aae5d104fbd83955d39ed1cfead75c0851f8` |
| `research/ecsc_hca_r0/r9_external_intake.py` | `1d386bd108be685c7d5d345774d6af9dcb0c260e387203b17d0acd4f070700e6` |
| `research/ecsc_hca_r0/run_dev.py` | `a2d2bc9fbc81b95bec789b38c5a879d502cc79d8db1f42b79f7c0311877e23bf` |
| `research/ecsc_hca_r0/run_r1.py` | `ee7d03396533b84c452df67c06c2f387f3dbaf6233a657528c985e2b51e1a5a6` |
| `research/ecsc_hca_r0/task.py` | `6bb07d65cf9bd958b23cbbe2d4fe0118e02cb076424ea14308f7643b87f0892f` |
| `research/ecsc_hca_r0/test_b2.py` | `1c550089811782db02579a6e8d24b7b6a3dfefc954451deaa86fabe5dba240d7` |
| `research/ecsc_hca_r0/test_baselines.py` | `71af19431a91e267a473be9071f0529b14e1b2942316f383c94c33aead8973ce` |
| `research/ecsc_hca_r0/test_r10_blind_retrieval.py` | `56e19f7ece02e2c081ec7e33550bef8e6eb2f8a9997eb4ac9fc8a1ac44383311` |
| `research/ecsc_hca_r0/test_r11_candidate_graph.py` | `6835f11b67fa248c5f83accc63382375d2e711e3a3d382ea74edcea582758af7` |
| `research/ecsc_hca_r0/test_r11_paired_audit.py` | `3be5b1b70634b58743cda33b34fa3952d367d3be8ea2e4038c6fd22551cd55a3` |
| `research/ecsc_hca_r0/test_r12_answer_level.py` | `9f87ead4965b36f738edd5154479d58d710b873b616a223659c5a8bfed7b04f5` |
| `research/ecsc_hca_r0/test_r12_native_structure.py` | `dac61763cfb2c7e07e50cabbd0ecf147b0237775f4c4c48f307b283bca1e7d2f` |
| `research/ecsc_hca_r0/test_r13_ingress_null.py` | `f72ad7ebbd87e0f05bc37d48c55f72fdcb7da4dce67f2d9c9fce8ac69320878c` |
| `research/ecsc_hca_r0/test_r14_native_projector.py` | `078499877b73760d62903419cbd46a2a6e8eb425bf4de8bde64f7c31c95297a9` |
| `research/ecsc_hca_r0/test_r15_candidate_ingress.py` | `2dbb40768a5b9dc440feed9dc5b74de471344c58cece68f7a7ec3691b19c6e58` |
| `research/ecsc_hca_r0/test_r16_scope_learning.py` | `3afdeb54d05c65b68ac26a5e26fbc2ca4e26219150f31f89657e59849ae19a00` |
| `research/ecsc_hca_r0/test_r17_raw_scope.py` | `6cbf567c1d1c629680a784c0eee1eba358b78c6cdac86f66200a9d90c517b803` |
| `research/ecsc_hca_r0/test_r17_scope_ablation_audit.py` | `76228978b82b2150030f6b3275528991693500aa21f83578ce728c02a382521a` |
| `research/ecsc_hca_r0/test_r18_bracket_binding.py` | `0a51b32c94fa70d0e93e1c36b6ebd97bd6f4f1e360873e0f685ddcdd38e913ac` |
| `research/ecsc_hca_r0/test_r18_shortcut_audit.py` | `1556da48bdbc37d7973f7206b5c8eb101183983b257cc29de23c92229115263b` |
| `research/ecsc_hca_r0/test_r19_fixed_root.py` | `abc1b4a6f9c14fe37ea135710721391b555c96ef4b0f0ca0e078549bd5a85c76` |
| `research/ecsc_hca_r0/test_r2_mechanics.py` | `cc76ccfb0f7cdf8ca2ca6da0a68340da91828a5340c92bc682331646b7c43389` |
| `research/ecsc_hca_r0/test_r3_warm_exact.py` | `937c94c1fffee310f7c23a5bf39cb92c1e1787f1d884a6d425c310df7284c18d` |
| `research/ecsc_hca_r0/test_r4_hacf_sources.py` | `6b7f6607be12803cae85a9e26a08a5bf480c30d6f337f5077e82d67c48ddf0e6` |
| `research/ecsc_hca_r0/test_r5_warm_publication.py` | `d151930f27c7315e436e40b07164d07096a9f342167a6735a35812683203f8c8` |
| `research/ecsc_hca_r0/test_r6_utility_probe.py` | `e39b3b0242de7d68c0293e4dd65ce19bcc17f6e8c0d5a2ffee99db672b857c4f` |
| `research/ecsc_hca_r0/test_r7_task_bound.py` | `df059386305eeb4ca794b2326636beb535331a7e5184c901a34dc15a03d2252a` |
| `research/ecsc_hca_r0/test_r8_accounting.py` | `9b77a337134794800fe59ccdb24a827a12910e1d2de8f24fcc742e91ceef115b` |
| `research/ecsc_hca_r0/test_r8_falsification.py` | `4ad5105f8dbb7aa6c4822b0ddcc3184b372fa2eb64d267a6e07d682b350df0a4` |
| `research/ecsc_hca_r0/test_r9_external_intake.py` | `218f9b26b17fec136d5713265eeb258b95a3e46b971b59da725584ba4d163b9c` |
| `research/ecsc_hca_r0/test_task.py` | `afef835794b62242ee794e269d72c869246f1d1e74b7fdb6510dfba510718256` |
| `research/ecsc_hca_r0/verify_r2_local.py` | `452fd3ef2b2d25870788142829d0b08c103d8ae15af866e55cf26e4392b37641` |
