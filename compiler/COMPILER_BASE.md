# TileLang-Ascend Compiler Base

- Repository: `https://github.com/tile-ai/tilelang-ascend.git`
- Base commit: `173de270512121f4208ffd77b8743371bd6e046d`
- Container source: `/workspace/tilelang-ascend`
- Local modifications: 6 files, 25 insertions, 2 deletions

Apply `tilelang_ascend_local.patch` at the repository root with `git apply`.
The `compiler_modified/` directory contains the same six post-patch files as a recovery snapshot.

Modified files:

- `src/op/ascend.cc`
- `src/op/ascend.h`
- `src/target/codegen_ascend.cc`
- `src/transform/allocate_tmp_buffer.cc`
- `src/transform/common/operation_config.h`
- `tilelang/language/ascend_tile.py`
